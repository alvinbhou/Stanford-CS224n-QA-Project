# SQuad With Transformer by HuggingFace

## Setup
Run the default setup first! See [READEME_bidaf](README_bidaf.md)

## SQuAD

Based on the script [`run_squad.py`](https://github.com/huggingface/transformers/blob/master/examples/run_squad.py).


### Benchmarks
| Model           | EM       | F1      | NoAns_f1 | HasAns_f1 | batch_size | epochs |
| --------------- | -------- | ------- | -------- | --------- | ---------- |--------|
| BERT-base       | 73.46    | 76.67   | 72.92    | 80.76     | 8          | 2      |
| BERT-large      | 81.56    | 84.92   | 84.91    | 84.92     | 8          | 2      |
| RoBERTa-base    | 78.82    | 82.16   | 80.93    | 83.49     | 8          | 2      |
| RoBERTa-large   | 82.82    | 86.32   | 87.59    | 84.94     | 4          | 2      |
| RoBERTa-large   | 83.35    | 86.62   | 86.36    | 86.43     | 6          | 3      |
| ALBERT-base-v2  | 78.38    | 81.50   | 81.50    | 81.51     | 8          | 2      |
| ALBERT-large-v2 | 81.16    | 84.22   | 83.24    | 85.29     | 6          | 2      |
|ALBERT-xxlarge-v1| 86.23    | 89.23   | 87.15    | 90.63     | 4          | 3     |


### Benchmarks For CLS Model
| Model           | EM       | F1      | NoAns_f1 | HasAns_f1 | batch_size | epochs |
| --------------- | -------- | ------- | -------- | --------- | ---------- |--------|
| Roberta-base    | 79.04    | 82.24   | 82.17    | 82.33     | 8          | 2      |
| ALBERT-base-v2  | 78.91    | 82.14   | 81.91    | 82.38     | 8          | 2      |


#### Training
For SQuAD2.0 example, you could run `./run_squad.sh`

For BERT-base on 8GB RAM GPU

```bash
python run_squad.py \
  --name bert-test-1 \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file data/train-v2.0.json \
  --predict_file data/dev-v2.0.json \
  --per_gpu_train_batch_size 8 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --version_2_with_negative \
  --evaluate_during_saving \
  --save_best_only \
  --logging_steps 500 \
  --save_steps 5000
```


For BERT-large on 12GB RAM GPU
```bash
python run_squad.py \
  --name bert-large \
  --model_type bert \
  --model_name_or_path bert-large-uncased-whole-word-masking-finetuned-squad \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file data/train-v2.0.json \
  --predict_file data/dev-v2.0.json \
  --per_gpu_train_batch_size 8 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 256 \
  --doc_stride 128 \
  --version_2_with_negative \
  --evaluate_during_saving \
  --save_best_only \
  --logging_steps 100 \
  --save_steps 4000
```

For RoBERTa-base on 8GB RAM GPU
```bash
python run_squad.py \
  --name roberta-base \
  --model_type roberta \
  --model_name_or_path roberta-base \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file data/train-v2.0.json \
  --predict_file data/dev-v2.0.json \
  --per_gpu_train_batch_size 8 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --version_2_with_negative \
  --evaluate_during_saving \
  --save_best_only \
  --logging_steps 100 \
  --save_steps 2000
```

For RoBERTa-large on 12GB RAM GPU
```bash
python run_squad.py \
  --name roberta-large \
  --model_type roberta \
  --model_name_or_path roberta-large \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file data/train-v2.0.json \
  --predict_file data/dev-v2.0.json \
  --per_gpu_train_batch_size 6 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --max_seq_length 256 \
  --doc_stride 128 \
  --seed 91 \
  --version_2_with_negative \
  --evaluate_during_saving \
  --save_best_only \
  --logging_steps 100 \
  --save_steps 3000
```

For ALBERT-base-v2 on 8GB RAM GPU
```bash
python run_squad \
  --name albert-base-v2 \
  --model_type albert \
  --model_name_or_path albert-base-v2 \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file data/train-v2.0.json \
  --predict_file data/dev-v2.0.json \
  --per_gpu_train_batch_size 8 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --version_2_with_negative \
  --evaluate_during_saving \
  --save_best_only \
  --logging_steps 50 \
  --save_steps 2000
```

For ALBERT-large-v2 on 12GB RAM GPU 
```bash
python run_squad.py \
  --name albert-large-v2 \
  --model_type albert \
  --model_name_or_path albert-large-v2 \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file data/train-v2.0.json \
  --predict_file data/dev-v2.0.json \
  --per_gpu_train_batch_size 6 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --version_2_with_negative \
  --evaluate_during_saving \
  --save_best_only \
  --logging_steps 50 \
  --save_steps 2000
```

For ALBERT-xlarge-v2 on 12GB RAM GPU (＃TODO, takes 48 hours to train)
```bash
python run_squad.py \
  --name albert-xlarge-v2 \
  --model_type albert \
  --model_name_or_path albert-xlarge-v2 \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file data/train-v2.0.json \
  --predict_file data/dev-v2.0.json \
  --per_gpu_train_batch_size 4 \
  --learning_rate 3e-5 \
  --num_train_epochs 3.0 \
  --max_seq_length 256 \
  --doc_stride 128 \
  --version_2_with_negative \
  --evaluate_during_saving \
  --save_best_only \
  --logging_steps 100 \
  --save_steps 3000
```

For ALBERT-xxlarge-v1 on NC24 (22hr per epoch)
```bash
python run_squad.py   
  --name albert-xxlarge-v1 \
  --model_type albert \
  --model_name_or_path albert-xxlarge-v1 \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file data/train-v2.0.json \
  --predict_file data/dev-v2.0.json \
  --per_gpu_train_batch_size 4  \
  --learning_rate 3e-5  \
  --num_train_epochs 3.0 \
  --max_seq_length 256 \
  --doc_stride 128 \
  --version_2_with_negative \
  --evaluate_during_saving  \
  --save_best_only \
  --logging_steps 100 \
  --save_steps 2000

```

For XLNet-base-cased on 12GB RAM K80 (currently not working)
```bash
python run_squad.py \
  --name xlnet-test \
  --model_type xlnet \
  --model_name_or_path xlnet-base-cased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file data/train-v2.0.json \
  --predict_file data/dev-v2.0.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --version_2_with_negative \
  --evaluate_during_saving \
  --save_best_only \
  --logging_steps 500 \
  --save_steps 5000
```

### Run CLS model
```bash
python run_squad_cls.py \
  --name bert-base-cls \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file data/train-v2.0.json \
  --predict_file data/dev-v2.0.json \
  --per_gpu_train_batch_size 4 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --version_2_with_negative \
  --evaluate_during_saving \
  --save_best_only \
  --logging_steps 100 \
  --save_steps 3000 \
  --seed 123 \
  --gradient_accumulation_steps 2
```


#### Dev Testing
Test for run_squad.py

```bash
python run_squad.py \
  --name bert-base-test \
  --model_type bert \
  --model_name_or_path save/train/bert-base-test-01 \
  --do_eval \
  --do_lower_case \
  --predict_file data/dev-v2.0.json \
  --per_gpu_train_batch_size 8 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --version_2_with_negative
```

Test for run_squad_cls.py
```bash
python run_squad_cls.py \
  --name bert-base-cls \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --eval_dir save/train/bert-base-cls-01/cur_best \
  --do_eval \
  --do_lower_case \
  --predict_file data/dev-v2.0.json \
  --per_gpu_train_batch_size 8 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --version_2_with_negative
```

#### Generate model outputs
```bash
python run_squad.py \
  --name albert-xxlarge-v1 \
  --model_type albert \
  --model_name_or_path save/train/albert-xxlarge-v1/cur_best \
  --do_output \
  --do_lower_case \
  --train_file data/train-v2.0.json \
  --predict_file data/dev-v2.0.json \
  --per_gpu_train_batch_size 8 \
  --max_seq_length 256 \
  --doc_stride 128 \
  --version_2_with_negative \
  --per_gpu_eval_batch_size 128
```

#### Run ensemble

```bash
python ensemble_squad.py \
 --name ensemble-test \
  --model_type placeholder \
  --model_name_or_path placeholder \
  --do_output \
  --do_lower_case \
  --train_file data/train-v2.0.json \
  --per_gpu_train_batch_size 8 \
  --max_seq_length 256 \
  --doc_stride 128 \
  --version_2_with_negative \
  --predict_file data/dev-v2.0.json \
  --saved_processed_data_dir save/output
```


### Ensemble features
```
albert-xxlarge-v1
albert-large-v2
roberta-large
bert-large
Shape: train (130319, 4, 2, 256) dev (6078, 4, 2, 256)
```

### Original Training script

#### Fine-tuning BERT on SQuAD1.0

This example code fine-tunes BERT on the SQuAD1.0 dataset. It runs in 24 min (with BERT-base) or 68 min (with BERT-large)
on a single tesla V100 16GB. The data for SQuAD can be downloaded with the following links and should be saved in a
$SQUAD_DIR directory.

* [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
* [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)
* [evaluate-v1.1.py](https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py)

And for SQuAD2.0, you need to download:

- [train-v2.0.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json)
- [dev-v2.0.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json)
- [evaluate-v2.0.py](https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/)

Below is the example of training BERT for SQuAD1.1

```bash
export SQUAD_DIR=/path/to/SQUAD

python run_squad.py \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file $SQUAD_DIR/train-v1.1.json \
  --predict_file $SQUAD_DIR/dev-v1.1.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /tmp/debug_squad/
```

Training with the previously defined hyper-parameters yields the following results on Squad 1.1:

```bash
f1 = 88.52
exact_match = 81.22
```

#### Distributed training


Here is an example using distributed training on 8 V100 GPUs and Bert Whole Word Masking uncased model to reach a F1 > 93 on SQuAD1.1:

```bash
python -m torch.distributed.launch --nproc_per_node=8 ./examples/run_squad.py \
    --model_type bert \
    --model_name_or_path bert-large-uncased-whole-word-masking \
    --do_train \
    --do_eval \
    --do_lower_case \
    --train_file $SQUAD_DIR/train-v1.1.json \
    --predict_file $SQUAD_DIR/dev-v1.1.json \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ./examples/models/wwm_uncased_finetuned_squad/ \
    --per_gpu_eval_batch_size=3   \
    --per_gpu_train_batch_size=3   \
```

Training with the previously defined hyper-parameters yields the following results:

```bash
f1 = 93.15
exact_match = 86.91
```

This fine-tuned model is available as a checkpoint under the reference
`bert-large-uncased-whole-word-masking-finetuned-squad`.

#### Fine-tuning XLNet on SQuAD

This example code fine-tunes XLNet on both SQuAD1.0 and SQuAD2.0 dataset. See above to download the data for SQuAD .

##### Command for SQuAD1.0:

```bash
export SQUAD_DIR=/path/to/SQUAD

python run_squad.py \
    --model_type xlnet \
    --model_name_or_path xlnet-large-cased \
    --do_train \
    --do_eval \
    --do_lower_case \
    --train_file $SQUAD_DIR/train-v1.1.json \
    --predict_file $SQUAD_DIR/dev-v1.1.json \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ./wwm_cased_finetuned_squad/ \
    --per_gpu_eval_batch_size=4  \
    --per_gpu_train_batch_size=4   \
    --save_steps 5000
```

##### Command for SQuAD2.0:

```bash
export SQUAD_DIR=/path/to/SQUAD

python run_squad.py \
    --model_type xlnet \
    --model_name_or_path xlnet-large-cased \
    --do_train \
    --do_eval \
    --version_2_with_negative \
    --train_file $SQUAD_DIR/train-v2.0.json \
    --predict_file $SQUAD_DIR/dev-v2.0.json \
    --learning_rate 3e-5 \
    --num_train_epochs 4 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ./wwm_cased_finetuned_squad/ \
    --per_gpu_eval_batch_size=2  \
    --per_gpu_train_batch_size=2   \
    --save_steps 5000
```

Larger batch size may improve the performance while costing more memory.

##### Results for SQuAD1.0 with the previously defined hyper-parameters:

```python
{
"exact": 85.45884578997162,
"f1": 92.5974600601065,
"total": 10570,
"HasAns_exact": 85.45884578997162,
"HasAns_f1": 92.59746006010651,
"HasAns_total": 10570
}
```

##### Results for SQuAD2.0 with the previously defined hyper-parameters:

```python
{
"exact": 80.4177545691906,
"f1": 84.07154997729623,
"total": 11873,
"HasAns_exact": 76.73751686909581,
"HasAns_f1": 84.05558584352873,
"HasAns_total": 5928,
"NoAns_exact": 84.0874684608915,
"NoAns_f1": 84.0874684608915,
"NoAns_total": 5945
}
```
