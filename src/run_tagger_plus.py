import os
import sys
import logging
import collections
from tqdm.auto import tqdm
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Union, Dict, Any

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, MSELoss
import datasets
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    set_seed,
    BertPreTrainedModel,
    BertModel,
    BertLayer
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from trainer import QuestionAnsweringTrainer
from eval_script import *

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/question-answering/requirements.txt")
logger = logging.getLogger(__name__)

os.environ["WANDB_DISABLED"] = "true"


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class TaggerPlusForMultiSpanQA(BertPreTrainedModel):
    def __init__(self, config, structure_lambda, span_lambda):
        super().__init__(config, structure_lambda, span_lambda)
        self.structure_lambda = structure_lambda
        self.span_lambda = span_lambda
        self.label2id= config.label2id
        self.num_labels = config.num_labels
        self.max_spans = 21
        self.max_pred_spans = 30
        self.H = config.hidden_size

        self.dense = nn.Linear(self.H, self.H)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.H, config.num_labels)
        self.num_span_outputs = nn.Sequential(nn.Linear(self.H, 64),nn.ReLU(),nn.Linear(64, 1))
        self.structure_outputs = nn.Sequential(nn.Linear(self.H, 128),nn.ReLU(),nn.Linear(128, 6))

        config.num_attention_heads=6 # for span encoder
        intermediate_size=1024
        self.span_encoder = BertLayer(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        num_span=None,
        structure=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs[0]
        pooled_output = outputs[1]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        # gather and pool span hidden
        B = pooled_output.size(0) # batch_size
        pred_spans = torch.zeros((B, self.max_pred_spans, self.H)).to(logits)
        pred_spans[:,0,:] = pooled_output # init the cls token use the bert cls token
        span_mask = torch.zeros((B, self.max_pred_spans)).to(logits)
        pred_labels = torch.argmax(logits, dim=-1)

        for b in range(B):
            s_pred_labels = pred_labels[b]
            s_sequence_output = sequence_output[b]
            indexes = [[]]
            flag=False
            for i in range(len(s_pred_labels)):
                if s_pred_labels[i] == self.label2id['B']: # B
                    indexes.append([i])
                    flag=True
                if s_pred_labels[i] == self.label2id['I'] and flag: # I
                    indexes[-1].append(i)
                if s_pred_labels[i] == self.label2id['O']: # O
                    flag=False
            indexes = indexes[:self.max_pred_spans]

            for i,index in enumerate(indexes):
                if i == 0:
                    span_mask[b,i] = 1
                    continue
                s_span = s_sequence_output[index[0]:index[-1]+1,:]
                s_span = torch.mean(s_span, dim=0) # mean pooling
                pred_spans[b,i,:] = s_span
                span_mask[b,i] = 1

        # encode span
        span_mask = span_mask[:,None,None,:] # extend for attention
        span_x = self.span_encoder(pred_spans, span_mask)[0]
        pooled_span_cls = span_x[:,0]
        pooled_span_cls = torch.tanh(self.dense(pooled_span_cls))

        num_span_logits = self.num_span_outputs(pooled_span_cls)
        structure_logits = self.structure_outputs(pooled_span_cls)

        outputs = (logits, num_span_logits, ) + outputs[:]
        if labels is not None: # for train
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)

            # num_span regression
            loss_mse = MSELoss()
            num_span=num_span.type(torch.float) / self.max_spans
            num_span_loss = loss_mse(num_span_logits.view(-1), num_span.view(-1))
            num_span_loss *= self.span_lambda
            # structure classification
            loss_focal = FocalLoss(gamma=0.5)
            structure_loss = loss_focal(structure_logits.view(-1, 6), structure.view(-1))
            structure_loss *= self.structure_lambda
            loss = loss + num_span_loss + structure_loss

            outputs = (loss, ) + outputs

        return outputs


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(default=False)
    structure_lambda: float= field(default=0.02)
    span_lambda: float= field(default=1)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(default="ner", metadata={"help": "The name of the task (ner, pos...)."})
    data_dir: Optional[str] = field(
        default=None, metadata={"help": "The dir of the dataset to use."}
    )
    train_file: Optional[str] = field(
        default='train.json', metadata={"help": "The dir of the dataset to use."}
    )
    question_column_name: Optional[str] = field(
        default='question', metadata={"help": "The column name of text to input in the file (a csv or JSON file)."}
    )
    context_column_name: Optional[str] = field(
        default='context', metadata={"help": "The column name of text to input in the file (a csv or JSON file)."}
    )
    label_column_name: Optional[str] = field(
        default='label', metadata={"help": "The column name of label to input in the file (a csv or JSON file)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_num_span: int = field(
        default=None,
    )
    max_seq_length: int = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. If set, sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    doc_stride: int = field(
        default=128,
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether to put the label for one word on all tokens of generated by that word or just on the "
            "one (in which case the other tokens will have a padding index)."
        },
    )
    save_embeds: bool = field(
        default=False,
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
    )
    max_eval_samples: Optional[int] = field(
        default=None,
    )
    max_predict_samples: Optional[int] = field(
        default=None,
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )


def main():
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    data_files = {'train': os.path.join(data_args.data_dir, data_args.train_file),
                  'validation':os.path.join(data_args.data_dir, "valid.json")}
    if training_args.do_predict:
                  data_files['test'] = os.path.join(data_args.data_dir, "test.json")
    raw_datasets = load_dataset('json', field='data', data_files=data_files)

    column_names = raw_datasets["train"].column_names
    features = raw_datasets["train"].features

    question_column_name = data_args.question_column_name
    context_column_name = data_args.context_column_name
    label_column_name = data_args.label_column_name

    structure_list = ['Complex', 'Conjunction', 'Non-Redundant', 'Redundant', 'Share', '']
    structure_to_id = {l: i for i, l in enumerate(structure_list)}

    label_list = ["B", "I", "O"]
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    # Map that sends B-Xxx label to its I-Xxx counterpart
    b_to_i_label = []
    for idx, label in enumerate(label_list):
        b_to_i_label.append(label_list.index(label.replace("B", "I")))

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    tokenizer_name_or_path = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    if config.model_type in {"gpt2", "roberta"}:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            add_prefix_space=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    model = TaggerPlusForMultiSpanQA.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        structure_lambda=model_args.structure_lambda,
        span_lambda=model_args.span_lambda,
    )

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet this "
            "requirement"
        )

    # Preprocessing the datasets.
    # Preprocessing is slighlty different for training and evaluation.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    else:
        column_names = raw_datasets["test"].column_names
    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # Training preprocessing
    def prepare_train_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name],
            examples[context_column_name],
            truncation="only_second",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=padding,
            is_split_into_words=True,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["labels"] = []
        tokenized_examples["num_span"] = []
        tokenized_examples["structure"] = []
        tokenized_examples["example_id"] = []
        tokenized_examples["word_ids"] = []
        tokenized_examples["sequence_ids"] = []

        for i, sample_index in enumerate(sample_mapping):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            label = examples[label_column_name][sample_index]
            word_ids = tokenized_examples.word_ids(i)
            previous_word_idx = None
            label_ids = [-100] * token_start_index

            for word_idx in word_ids[token_start_index:]:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label2id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            tokenized_examples["labels"].append(label_ids)
            tokenized_examples["num_span"].append(float(label_ids.count(0))) # count num of B as num_spans
            tokenized_examples["structure"].append(structure_to_id[examples['structure'][sample_index] if 'structure' in examples else ''])
            tokenized_examples["example_id"].append(examples["id"][sample_index])
            tokenized_examples["word_ids"].append(word_ids)
            tokenized_examples["sequence_ids"].append(sequence_ids)
        return tokenized_examples


    if training_args.do_train or data_args.save_embeds:
        train_examples = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_examples = train_examples.select(range(data_args.max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_examples.map(
                prepare_train_features,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )


    if training_args.do_eval:
        eval_examples = raw_datasets["validation"]
        # Validation Feature Creation
        if data_args.max_eval_samples is not None:
            eval_examples = eval_examples.select(range(data_args.max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_examples.map(
                prepare_train_features,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        predict_examples = raw_datasets["test"]
        # Predict Feature Creation
        if data_args.max_predict_samples is not None:
            predict_examples = predict_examples.select(range(data_args.max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_examples.map(
                prepare_train_features,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    tmp_train_dataset = train_dataset.remove_columns(["example_id","word_ids","sequence_ids"])
    tmp_eval_dataset = eval_dataset.remove_columns(["example_id","word_ids","sequence_ids"])

    # Run without Trainer

    import math
    import random

    from torch.utils.data import DataLoader
    from accelerate import Accelerator
    from huggingface_hub import Repository
    from transformers import (
        CONFIG_MAPPING,
        MODEL_MAPPING,
        AdamW,
        SchedulerType,
        get_scheduler,
    )

    accelerator = Accelerator()
    data_collator = DataCollatorForTokenClassification(
        tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
    )

    train_dataloader = DataLoader(
        tmp_train_dataset, shuffle=True, collate_fn=data_collator, batch_size=training_args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        tmp_eval_dataset, collate_fn=data_collator, batch_size=training_args.per_device_eval_batch_size
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)

    # Use the device given by the `accelerator` object.
    device = accelerator.device
    model.to(device)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    training_args.max_train_steps = int(training_args.num_train_epochs * num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=training_args.max_train_steps,
    )

    # Train!
    total_batch_size = training_args.per_device_train_batch_size * accelerator.num_processes * training_args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
    progress_bar = tqdm(range(training_args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    for epoch in range(int(training_args.num_train_epochs)):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs[0]
            loss = loss / training_args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % training_args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

    # evaluate
    model.eval()
    all_p = []
    all_span_p = []
    all_struct_p = []
    for batch in tqdm(eval_dataloader):
        with torch.no_grad():
            _, p, span_p, _, _ = model(**batch)
            all_p.append(p.cpu().numpy())
            all_span_p.append(span_p.cpu().numpy())

    all_p = [i for x in all_p for i in x]
    all_span_p = np.concatenate(all_span_p)


    # Post processing
    features = eval_dataset
    examples = eval_examples
    if len(all_p) != len(features):
        raise ValueError(f"Got {len(all_p[0])} predictions and {len(features)} features.")

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()
    all_confs = collections.OrderedDict()
    all_nums = collections.OrderedDict()

    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]
        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            sequence_ids = features[feature_index]['sequence_ids']
            word_ids = features[feature_index]['word_ids']
            confs = [np.max(l) for l in all_p[feature_index]]
            logits = [np.argmax(l) for l in all_p[feature_index]]
            labels = [id2label[l] for l in logits]
            nums = all_span_p[feature_index]
            prelim_predictions.append(
                {
                    "nums": nums,
                    "confs": confs,
                    "logits": logits,
                    "labels": labels,
                    "word_ids": word_ids,
                    "sequence_ids": sequence_ids
                }
            )

        previous_word_idx = -1
        ignored_index = [] # Some example tokens will be disappear after tokenization.
        valid_labels = []
        valid_confs = []
        valid_nums = sum(list(map(lambda x: x['nums'], prelim_predictions)))
        for x in prelim_predictions:
            confs = x['confs']
            labels = x['labels']
            word_ids = x['word_ids']
            sequence_ids = x['sequence_ids']

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            for word_idx,label,conf in list(zip(word_ids,labels,confs))[token_start_index:]:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    continue
                # We set the label for the first token of each word.
                elif word_idx > previous_word_idx:
                    ignored_index += range(previous_word_idx+1,word_idx)
                    valid_labels.append(label)
                    valid_confs.append(str(conf))
                    previous_word_idx = word_idx
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    continue

        context = example["context"]
        for i in ignored_index[::-1]:
            context = context[:i] + context[i+1:]
        assert len(context) == len(valid_labels) == len(valid_confs)

        predict_entities = get_entities(valid_labels, context)
        predict_confs = get_entities(valid_labels, valid_confs)
        confidence = [x[0] for x in predict_confs]
        predictions = [x[0] for x in predict_entities]
        all_predictions[example["id"]] = predictions
        all_confs[example['id']] = confidence
        all_nums[example["id"]] = valid_nums


    # Evaluate on valid
    golds = read_gold(os.path.join(data_args.data_dir, "valid.json"))
    print(multi_span_evaluate(all_predictions, golds))
    # Span adjustment
    for key in all_predictions.keys():
        if len(all_predictions[key]) > math.ceil(all_nums[key]*21):
            confs = list(map(lambda x: max([float(y) for y in x.split()]), all_confs[key]))
            new_preds = sorted(zip(all_predictions[key],confs), key=lambda x: x[1], reverse=True)[:math.ceil(all_nums[key]*21)]
            new_preds = [x[0] for x in new_preds]
            all_predictions[key] = new_preds
    # Evaluate again
    print(multi_span_evaluate(all_predictions, golds))

if __name__ == "__main__":
    main()
