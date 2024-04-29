## MOAA on Code Translation Task

## Task Definition
Code translation aims to migrate legacy software from one programming language in a platform toanother. In CodeXGLUE, given a piece of Java (C#) code, the task is to translate the code into C# (Java) version. Models are evaluated by BLEU scores, accuracy (exactly match), and [CodeBLEU](https://arxiv.org/abs/2009.10297) scores.

### Dataset
The dataset is collected from several public repos, including [Lucene](http://lucene.apache.org/), [POI](http://poi.apache.org/), [JGit](https://github.com/eclipse/jgit/) and [Antlr](https://github.com/antlr/).

These projects are originally developed in Java and subsequently translated into C#.

### Data Format
The dataset is in the "dataset" folder. Each line of the files is a function, and the suffix of the file indicates the programming language.

### Data Statistics

Data statistics of the dataset are shown in the below table:

|       | #Examples |
| ----- | :-------: |
| Train |  10,300   |
| Dev   |    500    |
| Test  |   1,000   |

## Fine-tune Victim Models
Taking Java to C# translation as an example, if you want to translate from C# to Java, you just need to change the filename.

### CodeBERT
```shell
python run.py \
    --output_dir ./saved_models/CodeBERT \
	--do_train \
	--do_eval \
    --do_text \
	--model_type roberta \
	--model_name_or_path microsoft/codebert-base \
	--config_name roberta-base \
	--tokenizer_name roberta-base \
	--train_filename ./data/train.java-cs.txt.java,./data/train.java-cs.txt.cs \
	--dev_filename ./data/valid.java-cs.txt.java,./data/valid.java-cs.txt.cs \
    --test_filename ./data/test.java-cs.txt.java,./data/test.java-cs.txt.cs \
	--max_source_length 512 \
	--max_target_length 512 \
	--beam_size 5 \
	--train_batch_size 8 \
	--eval_batch_size 16 \
	--learning_rate lr=5e-5 \
	--train_steps 8 \
    --seed 123456 2>&1 | tee train.log
```

### GraphCodeBERT
```shell
    source=java
    target=cs
    lr=1e-4
    batch_size=8
    beam_size=5
    source_length=512
    target_length=512
    output_dir=saved_models/GraphCodeBERT/
    train_file=data/train.java-cs.txt.$source,data/train.java-cs.txt.$target
    dev_file=data/valid.java-cs.txt.$source,data/valid.java-cs.txt.$target
    epochs=8
    pretrained_model=microsoft/graphcodebert-base

    python finetune_graphcodebert.py \
    --do_train \
    --do_eval \
    --model_type roberta \
    --source_lang $source \
    --model_name_or_path $pretrained_model \
    --tokenizer_name microsoft/graphcodebert-base \
    --config_name microsoft/graphcodebert-base \
    --train_filename $train_file \
    --dev_filename $dev_file \
    --output_dir $output_dir \
    --max_source_length $source_length \
    --max_target_length $target_length \
    --beam_size $beam_size \
    --train_batch_size $batch_size \
    --eval_batch_size $batch_size \
    --learning_rate $lr \
    --num_train_epochs $epochs \
    --seed 123456 2>&1| tee train.log
```

### CodeT5
For codeT5 we use the finetuned checkpoints in huggingface([Salesforce/codet5-base-codexglue-translate-java-cs](https://huggingface.co/Salesforce/codet5-base-codexglue-translate-java-cs) and [Salesforce/codet5-base-codexglue-translate-cs-java](https://huggingface.co/Salesforce/codet5-base-codexglue-translate-cs-java)).

## MOAA
```shell
python attack.py \
    --source_lang java \
    --target_lang c-sharp \
    --data_file=./data/test.java-cs.txt.java,./data/test.java-cs.txt.cs \
    --saved_victim_model_path=./saved_models/CodeBERT/checkpoint-best-ppl/model.bin \
    --model_type=codebert \
    --attack_numbers  100 2>&1 | tee moaa.log
```




