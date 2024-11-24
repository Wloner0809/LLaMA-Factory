# import transformers, torch
# from transformers import AutoTokenizer, AutoModelForCausalLM

# hf_path = "/home/v-wangyu1/model-ckpt/llama3.1-8b-full-sft-glanchatv2/checkpoint-4000"

# # Load the tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained(hf_path)
# model = AutoModelForCausalLM.from_pretrained(hf_path)
# pipeline = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

# sequences = pipeline(
#     inputs="The weather is nice today",
#     max_new_tokens=1024,
#     num_return_sequences=3,
#     do_sample=True,
#     temperature=0.7,
#     top_k=50,
#     top_p=0.95,
#     pad_token_id=tokenizer.eos_token_id,
#     eos_token_id=tokenizer.eos_token_id,
# )
# print(sequences)

import json
import os
import re
from functools import partial
from typing import Optional, Union

import numpy as np
import torch
from accelerate import Accelerator
from datasets import Dataset, DatasetDict, IterableDataset, load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    DataCollatorWithPadding,
    GenerationConfig,
    PreTrainedTokenizer,
    ProcessorMixin,
)

from llamafactory.data.aligner import convert_alpaca, convert_sharegpt
from llamafactory.data.parser import DatasetAttr
from llamafactory.data.processors.unsupervised import (
    preprocess_unsupervised_dataset,
    print_unsupervised_dataset_example,
)
from llamafactory.data.template import Template, get_template_and_fix_tokenizer
from llamafactory.extras.constants import FILEEXT2TYPE
from llamafactory.extras.logging import get_logger
from llamafactory.hparams import DataArguments


logger = get_logger(__name__)


def _get_preprocessed_dataset(
    dataset: Optional[Union["Dataset", "IterableDataset"]],
    data_args: "DataArguments",
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
    is_eval: bool = True,
) -> Optional[Union["Dataset", "IterableDataset"]]:
    r"""
    Preprocesses the dataset, including format checking and tokenization.
    """
    if dataset is None:
        return None

    preprocess_func = partial(
        preprocess_unsupervised_dataset,
        template=template,
        tokenizer=tokenizer,
        processor=processor,
        data_args=data_args,
    )
    print_function = partial(print_unsupervised_dataset_example, tokenizer=tokenizer)
    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            desc="Running tokenizer on dataset",
        )

    dataset = dataset.map(
        preprocess_func,
        batched=True,
        batch_size=data_args.preprocessing_batch_size,
        remove_columns=column_names,
        **kwargs,
    )

    try:
        print("eval example:" if is_eval else "training example:")
        print_function(next(iter(dataset)))
    except StopIteration:
        raise RuntimeError("Cannot find valid samples, check `data/README.md` for the data format.")

    return dataset


def align_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
) -> Union["Dataset", "IterableDataset"]:
    r"""
    Aligned dataset:
        _prompt: [{"role": "user", "content": "..."}] * (2T - 1)
        _response: [{"role": "assistant", "content": "..."}] * N (N > 1 for ranking dataset)
        _system: "..."
        _tools: "...",
        _images: [],
        _videos: [],
    """
    if dataset_attr.formatting == "alpaca":
        convert_func = partial(convert_alpaca, dataset_attr=dataset_attr, data_args=data_args)
    else:
        convert_func = partial(convert_sharegpt, dataset_attr=dataset_attr, data_args=data_args)

    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            desc="Converting format of dataset",
        )

    return dataset.map(
        convert_func,
        batched=False,
        remove_columns=column_names,
        **kwargs,
    )


def _load_single_dataset(
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
) -> Union["Dataset", "IterableDataset"]:
    r"""
    Loads a single dataset and aligns it to the standard format.
    """
    logger.info("Loading dataset {}...".format(dataset_attr))
    data_path, data_name, data_dir, data_files = None, None, None, None
    if dataset_attr.load_from in ["hf_hub", "ms_hub"]:
        data_path = dataset_attr.dataset_name
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "script":
        data_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "file":
        data_files = []
        local_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        if os.path.isdir(local_path):  # is directory
            for file_name in os.listdir(local_path):
                data_files.append(os.path.join(local_path, file_name))
                if data_path is None:
                    data_path = FILEEXT2TYPE.get(file_name.split(".")[-1], None)
                elif data_path != FILEEXT2TYPE.get(file_name.split(".")[-1], None):
                    raise ValueError("File types should be identical.")
        elif os.path.isfile(local_path):  # is file
            data_files.append(local_path)
            data_path = FILEEXT2TYPE.get(local_path.split(".")[-1], None)
        else:
            raise ValueError("File {} not found.".format(local_path))

        if data_path is None:
            raise ValueError("Allowed file types: {}.".format(",".join(FILEEXT2TYPE.keys())))
    else:
        raise NotImplementedError("Unknown load type: {}.".format(dataset_attr.load_from))

    dataset = load_dataset(
        path=data_path,
        name=data_name,
        data_dir=data_dir,
        data_files=data_files,
        split=dataset_attr.split,
        streaming=(data_args.streaming and (dataset_attr.load_from != "file")),
        trust_remote_code=True,
    )

    if data_args.streaming and (dataset_attr.load_from == "file"):  # faster than specifying streaming=True
        dataset = dataset.to_iterable_dataset()  # TODO: add num shards parameter

    if dataset_attr.num_samples is not None and not data_args.streaming:
        target_num = dataset_attr.num_samples
        indexes = np.random.permutation(len(dataset))[:target_num]  # all samples should be included
        target_num -= len(indexes)
        if target_num > 0:
            expand_indexes = np.random.choice(len(dataset), target_num)
            indexes = np.concatenate((indexes, expand_indexes), axis=0)

        assert len(indexes) == dataset_attr.num_samples, "Sample num mismatched."
        dataset = dataset.select(indexes)
        logger.info("Sampled {} examples from dataset {}.".format(dataset_attr.num_samples, dataset_attr))

    if data_args.max_samples is not None:  # truncate dataset
        max_samples = min(data_args.max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))

    return align_dataset(dataset, dataset_attr, data_args)


def get_dataset(
    template: "Template",
    data_args: "DataArguments",
    dataset_attr: "DatasetAttr",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
):
    r"""
    Gets the train dataset and optionally gets the evaluation dataset.
    """
    # Load and preprocess dataset
    dataset = _load_single_dataset(dataset_attr, data_args)
    dataset = _get_preprocessed_dataset(dataset, data_args, template, tokenizer, processor, is_eval=True)

    dataset_dict = {}
    if dataset is not None:
        if data_args.streaming:
            dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=0)
        dataset_dict["train"] = dataset

    dataset_dict = DatasetDict(dataset_dict)
    dataset_module = {}
    if "train" in dataset_dict:
        dataset_module["train_dataset"] = dataset_dict["train"]

    return dataset_module


def run_generation(
    accelerator: "Accelerator",
    model: "AutoModelForCausalLM",
    dataloader: torch.utils.data.DataLoader,
    generation_config,
    gen_kwargs,
):
    model, dataloader = accelerator.prepare(model, dataloader)
    accelerator.wait_for_everyone()
    outputs = []
    for i, batch in enumerate(tqdm(dataloader, desc="Inference", total=len(dataloader))):
        unwrapped_model = accelerator.unwrap_model(model)
        with torch.inference_mode():
            generated_tokens = unwrapped_model.generate(
                **batch,
                generation_config=generation_config,
                **gen_kwargs,
            )
            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            generated_tokens = accelerator.gather_for_metrics(generated_tokens).cpu().tolist()
        generated_tokens = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        outputs.append(generated_tokens)
    return outputs


if __name__ == "__main__":
    hf_path = "/mnt/default/finetuned-model/llama3.1-8b-full-sft-glanchatv2/checkpoint-4000"
    output_dir = "/mnt/default/results/llama3.1-8b-full-sft-glanchatv2-ckpt4000-round1.jsonl"
    tokenizer = AutoTokenizer.from_pretrained(hf_path)
    tokenizer.padding_side = "left"
    processor = AutoProcessor.from_pretrained(hf_path)
    data_args = DataArguments(
        template="llama3",
        dataset="glanchatv2-question",
        dataset_dir="/mnt/default/data",
    )
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_attr = DatasetAttr(
        dataset_name="glanchatv2-question.jsonl",
        load_from="file",
        formatting="alpaca",
        split="train",
    )
    data_module = get_dataset(
        template=template, data_args=data_args, dataset_attr=dataset_attr, tokenizer=tokenizer, processor=processor
    )
    data = data_module["train_dataset"]
    data.set_format(type="torch", columns=["input_ids", "attention_mask"])
    data = data.map(batched=True, batch_size=16)
    data_collator = DataCollatorWithPadding(tokenizer)
    dataloader = torch.utils.data.DataLoader(data, batch_size=4, collate_fn=data_collator)
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=hf_path)
    model.eval()
    generation_config = GenerationConfig.from_pretrained(hf_path)
    gen_kwargs = {
        "eos_token_id": [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids,
        "pad_token_id": tokenizer.pad_token_id,
        "max_new_tokens": 1024,
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 0.95,
        "num_return_sequences": 8,
    }
    accelerator = Accelerator()
    outputs = run_generation(
        accelerator=accelerator,
        model=model,
        dataloader=dataloader,
        generation_config=generation_config,
        gen_kwargs=gen_kwargs,
    )
    out_json = []
    for multi_output in outputs:
        json_element = {
            "instruction": re.search(r"user\n\n(.*?)assistant\n\n", multi_output[0], re.DOTALL).group(1),
            "input": "",
        }
        responses = []
        for single_output in multi_output:
            output = re.search(r"assistant\n\n(.*)", single_output, re.DOTALL).group(1)
            responses.append(output)
        json_element["output"] = responses
        out_json.append(json_element)
    with open("output_dir", "w") as f:
        for each in out_json:
            f.write(json.dumps(each) + "\n")
