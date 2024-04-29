# MOAA on Clone Detection Task

## Task Definition
Given two codes as the input, the task is to do binary classification (0/1), where 1 stands for semantic equivalence and 0 for others. Models are evaluated by F1 score.

### Dataset
The dataset we use is [BigCloneBench](https://www.cs.usask.ca/faculty/croy/papers/2014/SvajlenkoICSME2014BigERA.pdf) and filtered following the paper [Detecting Code Clones with Graph Neural Network and Flow-Augmented Abstract Syntax Tree](https://arxiv.org/pdf/2002.08653.pdf).


### Data Format

1. dataset/data.jsonl is stored in jsonlines format. Each line in the uncompressed file represents one function. One row is illustrated below. 
   
    - func: the function

    - idx: index of the example

2. train.txt/valid.txt/test.txt provide examples, stored in the following format: idx1 idx2 label

### Data Statistics

Data statistics of the dataset are shown in the below table:

|       | #Examples |
| ----- | :-------: |
| Train |  901,208  |
| Dev   |  415,416  |
| Test  |  415,416  |


## Fine-tune Victim Models
### CodeBERT
```shell
python finetune_codebert.py \
    --output_dir=./saved_models/CodeBERT \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --do_train \
    --do_test
    --train_data_file=../dataset/train.txt \
    --eval_data_file=../dataset/valid.txt \
    --test_data_file=../dataset/test.txt \
    --epoch 2 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee train.log

```

### GraphCodeBERT
```shell
python finetune_graphcodebert.py \
    --output_dir=saved_models/GraphCodeBERT \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --do_train \
    --train_data_file=../dataset/train_sampled.txt \
    --eval_data_file=../dataset/valid_sampled.txt \
    --test_data_file=../dataset/test_sampled.txt \
    --epoch 2 \
    --code_length 384 \
    --data_flow_length 128 \
    --train_batch_size 14 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee train.log
```

### CodeT5
```shell
python finetune_codet5.py \
    --output_dir=./saved_models/CodeT5 \
    --model_type=codet5 \
    --config_name=Salesforce/codet5-base \
    --model_name_or_path=Salesforce/codet5-base \
    --do_train \
    --do_test
    --train_data_file=../dataset/train.txt \
    --eval_data_file=../dataset/valid.txt \
    --test_data_file=../dataset/test.txt \
    --epoch 2 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee train.log
```


## MOAA
```shell
python attack.py \
    --data_file=./dataset/test.jsonl \
    --saved_victim_model_path=./saved_models/CodeBERT/checkpoint-best-f1/model.bin \
    --model_type=codebert \
    --attack_numbers  100 2>&1 | tee moaa.log
```