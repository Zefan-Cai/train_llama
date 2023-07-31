from ast import Raise
from datasets import load_dataset, load_metric
import random
import json
import copy
import ast
from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    default_data_collator,
)
import torch
import h5py
import torch.nn as nn
import numpy as np
import logging
from datasets import load_metric,Dataset,concatenate_datasets
from collections import defaultdict
from os.path import join
from sklearn.metrics import accuracy_score, zero_one_loss, precision_score, recall_score, f1_score, hamming_loss, roc_auc_score
from torch.nn.functional import pad
from typing import Any, Callable, Dict, List, NewType
from torchmetrics import BLEUScore
from glob import glob
InputDataClass = NewType("InputDataClass", Any)
from collections.abc import Mapping

logger = logging.getLogger(__name__)
IGNORE_INDEX = -100


class RationaleDataset():
    def __init__(self, tokenizer, model_args, data_args, training_args, config):

        self.tokenizer = tokenizer
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.config = config
        self.label_token='[LABEL]'
        self.rationale_token='[RATIONALE]'

        if self.data_args.pad_to_max_length:
            self.padding = "max_length"
        else:
            self.padding = "longest"

        if self.data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        self.max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
        if self.data_args.dataset_name in ['maven']:
            self.train_dataset = Dataset.from_json(self.data_args.train_file)
            self.eval_dataset = Dataset.from_json(self.data_args.validation_file)
            self.predict_dataset = Dataset.from_json(self.data_args.test_file)

            self.train_dataset = self.train_dataset.shuffle(training_args.seed)
            self.eval_dataset = self.eval_dataset.shuffle(training_args.seed)
            self.predict_dataset = self.predict_dataset.shuffle(training_args.seed)

            print(f"debug: train_dataset {self.train_dataset}")

            self.train_dataset = self.train_dataset.map(
                self.preprocess_function,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Runing one tokenization"
            )
            # self.eval_dataset = self.train_dataset
            # self.predict_dataset = self.train_dataset
            self.eval_dataset = self.eval_dataset.map(
                self.preprocess_function,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Runing one tokenization"
            )
            self.predict_dataset = self.predict_dataset.map(
                self.preprocess_function,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Runing one tokenization"
            )
            self.train_dataset = self.train_dataset.map(
                self.preprocess_function_batched,
                batched=True,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Runing one tokenization"
            )
            self.eval_dataset = self.eval_dataset.map(
                self.preprocess_function_batched,
                batched=True,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Runing one tokenization"
            )
            self.predict_dataset = self.predict_dataset.map(
                self.preprocess_function_batched,
                batched=True,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Runing one tokenization"
            )

            # Split train and dev datasets
            # self.train_dataset = self.train_dataset["train"]
            # self.eval_dataset = self.eval_dataset["train"]
            # self.predict_dataset = self.predict_dataset["train"]

            print(f"length train dataset {self.train_dataset.num_rows}")
            print(f"length eval dataset {self.eval_dataset.num_rows}")
            print(f"length test dataset {self.predict_dataset.num_rows}")

        if training_args.do_train:
            if data_args.max_train_samples is not None:
                self.train_dataset = self.train_dataset.select(range(data_args.max_train_samples))
        if training_args.do_eval:
            if data_args.max_eval_samples is not None:
                self.eval_dataset = self.eval_dataset.select(range(data_args.max_eval_samples))
        if training_args.do_predict or data_args.dataset_name is not None or data_args.test_file is not None:
            if data_args.max_predict_samples is not None:
                self.predict_dataset = self.predict_dataset.select(range(data_args.max_predict_samples))

        self.data_collator = self.collector
        self.metric = load_metric('accuracy')
        self.test_key = "accuracy"

    def preprocess_function_batched(self, examples):
        result = self.tokenizer(examples["label"], return_tensors="pt",padding=self.padding, max_length=self.max_seq_length, truncation=True)
        rationale_result= self.tokenizer(examples["rationale"],return_tensors="pt", padding=self.padding, max_length=self.max_seq_length, truncation=True)

        label_input = self.tokenizer(examples["label_input_text"], return_tensors="pt",padding=self.padding, max_length=self.max_seq_length, truncation=True)
        label_input_len = label_input.input_ids.ne(self.tokenizer.pad_token_id).sum(1)
        label_ids = copy.deepcopy(result.input_ids)
        for idx, source_len in enumerate(label_input_len):
            label_ids[idx,:source_len] = IGNORE_INDEX
        label_ids[~result.attention_mask.bool()]= IGNORE_INDEX

        rationale_input = self.tokenizer(examples["rationale_input_text"],  return_tensors="pt",padding=self.padding, max_length=self.max_seq_length, truncation=True)
        rationale_input_len = rationale_input.input_ids.ne(self.tokenizer.pad_token_id).sum(1)
        rationale_ids = copy.deepcopy(rationale_result.input_ids)
        for idx, source_len in enumerate(rationale_input_len):
            rationale_ids[idx,:source_len] = IGNORE_INDEX
        rationale_ids[~rationale_result.attention_mask.bool()]= IGNORE_INDEX

        result['rationale_input_ids'] = rationale_result["input_ids"]
        result['rationale_attention_mask'] = rationale_result["attention_mask"]
        result['label_ids'] = label_ids
        result['rationale_ids'] = rationale_ids
        result['gen_input_ids'] = label_input.input_ids
        result['gen_attention_mask'] = label_input.attention_mask
        result['gen_label_input_len'] = label_input_len
        
        return result

    def preprocess_function(self, examples):
        result = {}
        # result["input_text"] = examples["input_text"]
        result["label_input_text"] = " ".join([self.label_token , examples["prompt"],"Answer:"])
        result["rationale_input_text"] = " ".join([self.rationale_token , examples["prompt"],"Answer:"])
        result["label"] = " ".join([self.label_token, examples["prompt"], "Answer:",examples["label"]])
        result["rationale"] = " ".join([self.rationale_token , examples["prompt"], "Answer:",examples["rationale"][0][2:]])
        return result

    def collector(self, features: List[InputDataClass]):
        if not isinstance(features[0], Mapping):
            features = [vars(f) for f in features]
        first = features[0]
        batch = {}
        if "label_ids" in first and first["label_ids"] is not None:
            if isinstance(first["label_ids"], torch.Tensor):
                batch["labels"] = torch.stack([f["label_ids"] for f in features])
            else:
                dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
                batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)
            # batch["labels"][torch.where(batch["labels"]== self.tokenizer.pad_token_id)] = -100 # ignore_idx

        if "rationale_ids" in first and first["rationale_ids"] is not None:
            if isinstance(first["rationale_ids"], torch.Tensor):
                batch["rationales"] = torch.stack([f["rationale_ids"] for f in features])
            else:
                dtype = torch.long if type(first["rationale_ids"][0]) is int else torch.float
                batch["rationales"] = torch.tensor([f["rationale_ids"] for f in features], dtype=dtype)   
            # batch["rationales"][torch.where(batch["rationales"]== self.tokenizer.pad_token_id)] = -100 # ignore_idx
        ignored_keys = ['label_ids', 'rationale_ids','options','finish_reason','line_idx','token_type_ids']
        for k, v in first.items():
            if  v is not None and not isinstance(v, str) and k not in ignored_keys :
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                else:
                    batch[k] = torch.tensor([f[k] for f in features])

        # for each in batch:
        #     if isinstance(batch[each], torch.Tensor) and (batch[each].dtype == torch.float32):
        #         batch[each] = batch[each].to(dtype= torch.bfloat16)
        return batch

    def compute_metrics(self, p: EvalPrediction):

        preds = p.predictions
        labels = p.label_ids
        preds[preds==IGNORE_INDEX] = self.tokenizer.pad_token_id
        labels[labels==IGNORE_INDEX] = self.tokenizer.pad_token_id
        bleu = BLEUScore()
        bleu_result = []
        accuracy = 0
        dict_return={}
        for i,pred in enumerate(preds):
            p_token = self.tokenizer.decode(pred,skip_special_tokens=True)
            label_token = self.tokenizer.decode(labels[i],skip_special_tokens=True)
            # if self.training_args.multiple_choice:
            #     l = label_token.split(' ')[0]
            #     if l in p_token:
            #         accuracy+=1
            #     bleu_result.append(bleu([p_token],[[label_token]]).item())
            # else:
            #     if p_token == label_token:
            #         accuracy+=1
            if label_token in p_token:
                accuracy+=1

        # if self.training_args.multiple_choice:
        #     dict_return ={
        #     # "BleuScore": bleu_result,
        #     'Avg_BleuScore':np.array(bleu_result).mean(),
        #     }
        dict_return['accuracy'] = accuracy/len(preds)
        return dict_return
