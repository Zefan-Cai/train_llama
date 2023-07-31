import logging
import random
from typing import Dict
from transformers import AdamW,LlamaConfig,AutoTokenizer
import transformers
from .dataset import RationaleDataset
from model.utils import get_model
from training.trainer_llama import LlamaTrainer


import torch
logger = logging.getLogger(__name__)
DEFAULT_PAD_TOKEN="[PAD]"
DEFAULT_LABEL_TOKEN="[LABEL]"
DEFAULT_RATIONALE_TOKEN="[RATIONALE]"
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def get_trainer(args):
    model_args, data_args, training_args = args

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="right",
        use_fast=True,
    )
    # sp = ['[LABEL]','[RATIONALE]']
    # sp = sp+tokenizer.additional_special_tokens[len(sp):]
    # tokenizer.add_special_tokens({'additional_special_tokens':sp})
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    config = LlamaConfig.from_pretrained(
        model_args.model_name_or_path,

    )
    model = get_model(model_args, config)
    # model = get_model(model_args, config).to(dtype=torch.bfloat16,device=training_args.device)
    
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN,additional_special_tokens=[DEFAULT_LABEL_TOKEN,DEFAULT_RATIONALE_TOKEN]),
                tokenizer=tokenizer,
                model=model,
            )
    dataset = RationaleDataset(tokenizer, model_args, data_args, training_args, config)

    if training_args.do_train:
        for index in random.sample(range(len(dataset.train_dataset)), 1):
            logger.info(f"Sample keys {index} of the training set: {dataset.train_dataset[index].keys()}.")

            input_text = dataset.train_dataset[index]["prompt"]
            logger.info(f"Sample input_text {index} of the training set: {input_text}.")
            output_text = dataset.train_dataset[index]["rationale"]
            logger.info(f"Sample output_text {index} of the training set: {output_text}.")

            input_ids = dataset.train_dataset[index]["input_ids"]
            logger.info(f"Sample input_ids {index} of the training set: {input_ids}.")
            attention_mask = dataset.train_dataset[index]["attention_mask"]
            logger.info(f"Sample attention_mask {index} of the training set: {attention_mask}.")
            label = dataset.train_dataset[index]["label"]
            logger.info(f"Sample label {index} of the training set: {label}.")


    # model = get_model(model_args, config).to(dtype=torch.bfloat16,device=training_args.device)

    trainer = LlamaTrainer(
        tokenizer=tokenizer,
        model=model,
        config=config,
        args=training_args,
        model_args=model_args,
        train_dataset=dataset.train_dataset if training_args.do_train else None,
        # test_dataset=dataset.eval_dataset if training_args.do_test else None,
        eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
        predict_dataset=dataset.predict_dataset, 
        compute_metrics=dataset.compute_metrics,
        data_collator=dataset.data_collator,
    )

    return trainer, dataset.predict_dataset
