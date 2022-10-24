# MultiSpanQA: A Dataset for Multi-Span Question Answering

Welcome to submit your model's prediction to our [leaderboard](https://multi-span.github.io).

## Requirements

Python >= 3.7

pytorch >= 1.8.1

huggingface >= 4.17.0

## Fine-tuning BERT tagger on MultiSpanQA

```bash
python run_tagger.py \
    --model_name_or_path bert-base-uncased \
    --data_dir ../data/MultiSpanQA_data \
    --output_dir ../output \
    --overwrite_output_dir \
    --overwrite_cache \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 4 \
    --eval_accumulation_steps 50 \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --max_seq_length  512 \
    --doc_stride 128 
```
## Fine-tuning Huggingface QA model on MultiSpanQA

Since the QA model is single-span model, you need to change MultiSpanQA to the format that can be trained on single-span model by run:

```bash
python generate_squad_format.py
```

This will generate two train files in squad formet. You can choose to fine-tune BERT on one of them (for example v1) using:

```bash
python run_squad.py \
    --model_name_or_path bert-base-uncased \
    --train_file ../data/MultiSpan_data/squad_train_softmax_v1.json \
    --validation_file ../data/MultiSpan_data/squad_valid.json \
    --output_dir ../output \
    --overwrite_output_dir \
    --overwrite_cache \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 4 \
    --eval_accumulation_steps 50 \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --max_seq_length  512 \
    --doc_stride 128 

```
