# MOAA on Authorship Attribution Task

## Task Definition
The purpose of this task is to analyze coding patterns, stylistic features, and other idiosyncratic attributes in the source code to accurately predict the author’s identity.

### Dataset
The dataset we use comes from the paper [Source Code Authorship Attribution Using Long Short-Term Memory Based Networks](https://link.springer.com/chapter/10.1007/978-3-319-66402-6_6). It is collected from the Google Code Jam1 (GCJ), an annual competition held by Google since 2008. This dataset consists of solutions to 10 problems implemented by 70 authors.

### Data Statistics

Data statistics of the dataset are shown in the below table:

|       | #Examples |
| ----- | :-------: |
| Train |    528    |
| Eval  |    132    |


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
    --train_data_file=./dataset/train.txt \
    --eval_data_file=./dataset/valid.txt \
    --test_data_file=./dataset/valid.txt \
    --epoch 30 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
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
    --train_data_file=./dataset/train.txt \
    --eval_data_file=./dataset/valid.txt \
    --test_data_file=./dataset/valid.txt \
    --epoch 30 \
    --code_length 384 \
    --data_flow_length 128 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
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
    --train_data_file=./dataset/train.txt \
    --eval_data_file=./dataset/valid.txt \
    --test_data_file=./dataset/valid.txt \
    --cache_path=./dataset \
    --epoch 30 \
    --max_source_length 512 \
    --train_batch_size 8 \
    --eval_batch_size 16 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --seed 123456  2>&1 | tee train.log
```


## MOAA
```shell
python attack.py \
    --data_file=./dataset/valid.txt \
    --saved_victim_model_path=./saved_models/CodeBERT/checkpoint-best-f1/model.bin \
    --model_type=codebert \
    --attack_numbers  100 2>&1 | tee moaa.log
```
