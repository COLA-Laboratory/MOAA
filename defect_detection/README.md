# MOAA on Defect Detection Task

## Task Definition
Given a source code, the task is to identify whether it is an insecure code that may attack software systems, such as resource leaks, use-after-free vulnerabilities and DoS attack. We treat the task as binary classification (0/1), where 1 stands for insecure code and 0 for secure code.

### Dataset

The dataset we use comes from the paper [*Devign*: Effective Vulnerability Identification by Learning Comprehensive Program Semantics via Graph Neural Networks](http://papers.nips.cc/paper/9209-devign-effective-vulnerability-identification-by-learning-comprehensive-program-semantics-via-graph-neural-networks.pdf). We combine all projects and split 80%/10%/10% for training/dev/test.

### Download and Preprocess

1.Download dataset from [website](https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF/view?usp=sharing) to "dataset" folder or run the following command:

```shell
cd dataset
pip install gdown
gdown https://drive.google.com/uc?id=1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF
cd ..
```

2.Preprocess dataset

```shell
cd dataset
python preprocess.py
cd ..
```

### Data Format

After preprocessing dataset, you can obtain three .jsonl files, i.e. train.jsonl, valid.jsonl, test.jsonl

For each file, each line in the uncompressed file represents one function.  One row is illustrated below.

   - **func:** the source code
   - **target:** 0 or 1 (vulnerability or not)
   - **idx:** the index of example

### Data Statistics

Data statistics of the dataset are shown in the below table:

|       | #Examples |
| ----- | :-------: |
| Train |  21,854   |
| Dev   |   2,732   |
| Test  |   2,732   |


## Fine-tune Victim Models
### CodeBERT
```shell
python finetune_codebert.py \
    --output_dir=./saved_models/CodeBERT \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --do_train \
    --do_test \
    --train_data_file=./dataset/train.jsonl \
    --eval_data_file=./dataset/valid.jsonl \
    --test_data_file=./dataset/test.jsonl \
    --epoch 5 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log
```

### GraphCodeBERT
```shell
python finetune_graphcodebert.py \
    --output_dir=./saved_models/GraphCodeBERT \
    --model_type=roberta \
    --config_name=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --do_train \
    --do_test \
    --train_data_file=./dataset/train.jsonl \
    --eval_data_file=./dataset/valid.jsonl \
    --test_data_file=./dataset/test.jsonl \
    --epoch 5 \
    --code_length 384 \
    --data_flow_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1 | tee train.log
```

### CodeT5
```shell
CUDA_VISIBLE_DEVICES=1 python finetune_codet5.py \
    --output_dir=./saved_models/CodeT5 \
    --model_type=codet5 \
    --tokenizer_name=Salesforce/codet5-base \
    --model_name_or_path=Salesforce/codet5-base \
    --do_train \
    --do_eval \
    --do_test \
    --train_data_file=./dataset/train.jsonl \
    --eval_data_file=./dataset/valid.jsonl \
    --test_data_file=./dataset/valid.jsonl \
    --cache_path=./dataset \
    --epoch 5 \
    --max_source_length 512 \
    --train_batch_size 8 \
    --eval_batch_size 16 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --seed 123456  2>&1 | tee train.log
```

## MOAA
```shell
python attack.py \
    --data_file=./dataset/test.jsonl \
    --saved_victim_model_path=./saved_models/CodeBERT/checkpoint-best-acc/model.bin \
    --model_type=codebert \
    --attack_numbers  100 2>&1 | tee moaa.log
```