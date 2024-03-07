import os
import sys

sys.path.append("../")
sys.path.append("../parser")
from transformers import (
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    T5ForConditionalGeneration,
    T5EncoderModel,
    T5Config,
    T5Tokenizer,
)
from moaa import MOAA
from model import CodeBERTModel, GraphCodeBERTModel, CodeT5Model
import torch
import argparse
import logging
import json
from tqdm import tqdm
from utils import Example
import random
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def load_victim_model(model_type, saved_victim_path):
    if model_type == "codebert":
        config = RobertaConfig.from_pretrained("microsoft/codebert-base")
        config.num_labels = 1
        tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        encoder = RobertaForSequenceClassification.from_pretrained(
            "microsoft/codebert-base", config=config
        )
        model = CodeBERTModel(encoder, config, tokenizer)
        # model.load_state_dict(torch.load(saved_victim_path))
        model.to("cuda")
        logger.info("Loaded victim model from {}.".format(saved_victim_path))
    elif model_type == "graphcodebert":
        config = RobertaConfig.from_pretrained("microsoft/graphcodebert-base")
        config.num_labels = 1
        tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
        model = RobertaForSequenceClassification.from_pretrained(
            "microsoft/graphcodebert-base", config=config
        )
        model = GraphCodeBERTModel(model, config, tokenizer)
        model.load_state_dict(torch.load(saved_victim_path))
        model.to("cuda")
        logger.info("Loaded victim model from {}.".format(saved_victim_path))
    elif model_type == "codet5":
        config = T5Config.from_pretrained("Salesforce/codet5-base")
        tokenizer = T5Tokenizer.from_pretrained("Salesforce/codet5-base")
        model = T5ForConditionalGeneration.from_pretrained(
            "Salesforce/codet5-base", config=config
        )
        model = CodeT5Model(model, config, tokenizer)
        model.load_state_dict(torch.load(saved_victim_path))
        model.to("cuda")
        logger.info("Loaded victim model from {}.".format(saved_victim_path))
    else:
        raise ValueError(
            "Invalid model type: {}, select from {codebert, graphcodebert, codet5}".format(
                model_type
            )
        )
    return model, tokenizer


def main():
    # Load parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file",
        required=True,
        type=str,
        help="The path to the attacked data file",
    )
    parser.add_argument(
        "--saved_victim_model_path",
        required=True,
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--model_type",
        default="codebert",
        type=str,
        help="The type of the victim model",
    )

    parser.add_argument(
        "--attack_numbers",
        default=100,
        type=int,
        help="The number of examples to attack",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="The random seed for the attack",
    )
    parser.add_argument(
        "--csv_store_path",
        default="attack_results.csv",
        type=str,
        help="The path to store the attack results",
    )

    args = parser.parse_args()

    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # load victim model
    model, tokenizer = load_victim_model(args.model_type, args.saved_victim_model_path)

    # load T5 model
    codet5_path = "Salesforce/codet5-base"
    t5_tokenizer = RobertaTokenizer.from_pretrained(codet5_path)
    t5_model = T5ForConditionalGeneration.from_pretrained(codet5_path)
    model_dir = "./saved_models/assisted_model/pytorch_model.bin"
    t5_model.load_state_dict(torch.load(model_dir))
    t5_model.to("cuda")  # this model is for identifier prediction
    logger.info("Loaded CodeT5 model from{}.".format(model_dir))

    t5_config = T5Config.from_pretrained(codet5_path)
    t5_emb_model = T5EncoderModel.from_pretrained(codet5_path, config=t5_config)
    t5_emb_model.to("cuda")  # this model is for similarity calculation

    # Load data
    examples = []
    idx = 0
    with open(args.data_file) as f:
        for line in f:
            js = json.loads(line.strip())
            examples.append(
                Example(idx=idx, source=js["func"].strip(), target=js["target"])
            )
            idx += 1
    # sample the attacked examples
    examples = random.sample(examples, min(args.attack_numbers, len(examples)))

    # Initialize the attacker
    attacker = MOAA(
        args,
        model,
        tokenizer,
        t5_model,
        t5_tokenizer,
        t5_emb_model,
        pop_size=30,
    )

    if os.path.exists(args.csv_store_path) == False:
        columns = [
            "Index",
            "Original Code",
            "Program Length",
            "Adversarial Code",
            "True Label",
            "Original Prediction",
            "Adv Prediction",
            "Is Success",
            "Extracted Name",
            "Replaced Names",
            "Original Prob",
            "F1",
            "F2",
            "F3",
            "Query Times",
        ]
        df = pd.DataFrame(data=None, columns=columns)
        df.to_csv(args.csv_store_path)

    attack_results = []  # store the attack results
    pbar = tqdm(total=len(examples))
    for e in examples:
        is_success = attacker.attack(e.idx, e.source, e.target)
        attack_results.append(is_success)
        pbar.update()

    # Analyze the attack results
    total_cnt = 0
    success_cnt = 0
    for index in range(len(attack_results)):
        is_success = attack_results[index]
        if (
            is_success == 1 or is_success == -1
        ):  # the original example is attacked, i.e. the original example is correctly predicted and has the identifiers to be perturbed
            total_cnt += 1
        if is_success == 1:  # the attack is successful
            success_cnt += 1
        logger.info("example={}, is_success={}".format(index, is_success))
        logger.info(
            "*****total_num={}, success_num={}, fail_num={}, skip_num={}, success_rate={}*****".format(
                index + 1,
                success_cnt,
                total_cnt - success_cnt,
                index + 1 - total_cnt,
                float(success_cnt / (total_cnt + 1e-16)),
            )
        )

    logger.info("*****All Finished, ACC={}".format(success_cnt / (total_cnt + 1e-16)))


if __name__ == "__main__":
    main()
