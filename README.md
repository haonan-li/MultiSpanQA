# MultiSpanQA: A Dataset for Multi-Span Question Answering

Welcome to submit your model's prediction to our [leaderboard](https://multi-span.github.io).

## Requirements

Python >= 3.7

pytorch >= 1.8.1

huggingface >= 4.17.0

## Fine-tune BERT tagger on MultiSpanQA (Recommended)

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
To try other encoders, replace the model name `bert-base-uncased` with other model names, currently we support `bert-large-uncased`, `roberta-base` and `roberta-large`.
You are expected to get similar results as:

<table>
  <tr>
    <th>Encoder</th>
	<th colspan="3">Exact Match</th>
	<th colspan="3">Partial Match</th>
  </tr>
  <tr>
    <th></th>
    <th>Precision.</th>
    <th>Recall</th>
    <th>F1</th>
    <th>Precision.</th>
    <th>Recall</th>
    <th>F1</th>
  </tr>
  <tr>
    <th>BERT-base</th>
    <th>55.53</th>
    <th>63.51</th>
    <th>59.25</th>
    <th>76.71</th>
    <th>75.52</th>
    <th>76.11</th>
  </tr>
  <tr>
    <th>BERT-large</th>
    <th>59.25</th>
    <th>64.47</th>
    <th>61.75</th>
    <th>78.79</th>
    <th>77.24</th>
    <th>78.01</th>
  </tr>
  <tr>
    <th>Roberta-base</th>
    <th>61.43</th>
    <th>67.30</th>
    <th>64.23</th>
    <th>80.72</th>
    <th>79.83</th>
    <th>80.27</th>
  </tr>
  <tr>
    <th>Roberta-large</th>
    <th>66.02</th>
    <th>71.84</th>
    <th>68.81</th>
    <th>84.16</th>
    <th>84.61</th>
    <th>84.39</th>
  </tr>
</table>


## Fine-tune Huggingface QA model on MultiSpanQA

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
