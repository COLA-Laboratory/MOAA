# MOAA on Code Summarization Task

## Task Definition
The task is to generate natural language comments for a code, and evaluted by [smoothed bleu-4](https://aclanthology.org/C04-1072.pdf) score.

### Dataset
The dataset we use comes from [CodeSearchNet](https://arxiv.org/pdf/1909.09436) and we filter the dataset as the following:

Remove examples that codes cannot be parsed into an abstract syntax tree.
Remove examples that #tokens of documents is < 3 or >256
Remove examples that documents contain special tokens (e.g. <img ...> or https:...)
Remove examples that documents are not English.

### Download data and preprocess
```shell
unzip dataset.zip
cd dataset
wget https://zenodo.org/record/7857872/files/go.zip
wget https://zenodo.org/record/7857872/files/java.zip
wget https://zenodo.org/record/7857872/files/javascript.zip
wget https://zenodo.org/record/7857872/files/php.zip
wget https://zenodo.org/record/7857872/files/python.zip
wget https://zenodo.org/record/7857872/files/ruby.zip

unzip python.zip
unzip java.zip
unzip ruby.zip
unzip javascript.zip
unzip go.zip
unzip php.zip
rm *.zip
rm *.pkl

python preprocess.py
rm -r */final
cd ..
```

### Data Format
After preprocessing dataset, you can obtain three .jsonl files, i.e. train.jsonl, valid.jsonl, test.jsonl

For each file, each line in the uncompressed file represents one function. One row is illustrated below.

- repo: the owner/repo
- path: the full path to the original file
- func_name: the function or method name
- original_string: the raw string before tokenization or parsing
- language: the programming language
- code/function: the part of the original_string that is code
- code_tokens/function_tokens: tokenized version of code
- docstring: the top-level comment or docstring, if it exists in the original string
- docstring_tokens: tokenized version of docstring

### Data Statistic

Data statistics of the dataset are shown in the below table:
| Programming Language | Training |  Dev   |  Test  |
| -------------------- | :------: | :----: | :----: |
| Python               | 251,820  | 13,914 | 14,918 |
| PHP                  | 241,241  | 12,982 | 14,014 |
| Go                   | 167,288  | 7,325  | 8,122  |
| Java                 | 164,923  | 5,183  | 10,955 |
| JavaScript           |  58,025  | 3,885  | 3,291  |
| Ruby                 |  24,927  | 1,400  | 1,261  |

## Fine-tune Victim Models
### CodeBERT
```shell
lang=ruby #programming language
lr=5e-5
batch_size=16
beam_size=5
source_length=512
target_length=128
data_dir=./dataset
output_dir=saved_models/CodeBERT/$lang
train_file=$data_dir/$lang/train.jsonl
dev_file=$data_dir/$lang/valid.jsonl
test_flie=$data_dir/$lang/test.jsonl
epochs=10
pretrained_model=microsoft/codebert-base

python finetune_codebert.py \
    --do_train \
    --do_eval \
    --do_test \
    --model_type roberta \
    --model_name_or_path $pretrained_model \
    --train_filename $train_file \
    --dev_filename $dev_file \
    --test_filename $test_file \
    --output_dir $output_dir \
    --max_source_length $source_length \
    --max_target_length $target_length \
    --beam_size $beam_size \
    --train_batch_size $batch_size \
    --eval_batch_size $batch_size \
    --learning_rate $lr \
    --num_train_epochs $epochs
    --seed 123456 2>&1| tee train.log
```

### CodeT5
For codeT5 we use the finetuned checkpoints in [huggingface](https://huggingface.co/Salesforce).


## MOAA

```shell
python attack.py \
    --lang python \
    --data_file=./dataset/test.jsonl \
    --saved_victim_model_path=./saved_models/CodeBERT/checkpoint-best-ppl/model.bin \
    --model_type=codebert \
    --attack_numbers  100 2>&1 | tee moaa.log
```