python attack.py \
    --data_file=./dataset/test.jsonl \
    --saved_victim_model_path=./saved_models/checkpoint-best-acc/model.bin \
    --model_type=codebert \
    --attack_numbers  5 2>&1 | tee train.log
