import sys

sys.path.append("../")
sys.path.append("../parser")

from run_parser import get_identifiers
import logging
import numpy as np
import pandas as pd
from utils import get_identifier_posistions_from_code, Individual, Population
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class MOAA:
    def __init__(
        self, args, model, tokenizer, t5_model, t5_tokenizer, t5_emb_model, pop_size
    ):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.t5_model = t5_model
        self.t5_tokenizer = t5_tokenizer
        self.t5_emb_model = t5_emb_model
        self.pop_size = pop_size
        self.parent_pop = Population(self.pop_size)
        self.offspring_pop = Population(self.pop_size)
        self.mixed_pop = Population(self.pop_size * 2)
        self.model_type = args.model_type

    def attack(self, idx, source, target):
        self.parent_pop.indi = []
        self.offspring_pop.indi = []

        # calculate the original codebleu

        codebleu_score = 0.0

        query_times = 0
        flag_success = False  # whether we have found the first adversarial example in our algorithm
        is_success = 0
        identifiers, code_tokens = get_identifiers(source, self.args.lang)
        identifiers = [i[0] for i in identifiers]
        if len(identifiers) == 0:  # no identifier in the code
            is_success = -3
        else:  # begin to attack
            names_positions_dict = get_identifier_posistions_from_code(
                code_tokens, identifiers
            )
            pos_dict = []
            self.max_iteration = 5 * len(identifiers)
            for i in identifiers:
                pos_dict.append(names_positions_dict[i])

            # Initialze the population
            for i in range(self.pop_size):
                self.parent_pop.indi.append(
                    Individual(
                        code_tokens, identifiers, pos_dict, target, self.model_type
                    )
                )
                self.offspring_pop.indi.append(
                    Individual(
                        code_tokens, identifiers, pos_dict, target, self.model_type
                    )
                )
                self.parent_pop.indi[i].mutation(self.t5_model, self.t5_tokenizer)
                self.mixed_pop.indi.append(
                    Individual(
                        code_tokens, identifiers, pos_dict, target, self.model_type
                    )
                )
                self.mixed_pop.indi.append(
                    Individual(
                        code_tokens, identifiers, pos_dict, target, self.model_type
                    )
                )

                self.parent_pop.indi[i].function_eval(
                    self.model,
                    self.tokenizer,
                    self.t5_emb_model,
                    self.t5_tokenizer,
                )
                if flag_success == False:
                    query_times += 1
                    if (
                        self.parent_pop.indi[i].obj_[0] < 0.5 * codebleu_score
                    ):  # the first adversarial example is found
                        flag_success = True

            # Begin the evolution process
            for i in range(self.max_iteration):
                # crossover
                self.offspring_pop.crossover(self.parent_pop)
                # mutation
                self.offspring_pop.mutation(self.t5_model, self.t5_tokenizer)
                # evaluate the objectives of the offspring population
                for j in range(self.pop_size):
                    self.offspring_pop.indi[j].function_eval(
                        self.model,
                        self.tokenizer,
                        self.t5_emb_model,
                        self.t5_tokenizer,
                    )
                    if flag_success == False:
                        query_times += 1
                        if self.offspring_pop.indi[j].obj_[0] < 0.5 * codebleu_score:
                            flag_success = True
                # environmental selection
                self.parent_pop.environmental_selection(
                    self.offspring_pop, self.mixed_pop
                )

        # attack finished, save the results
        if is_success == -3:
            res = [
                [
                    idx,
                    source,
                    len(source),
                    None,
                    None,
                    target,
                    None,
                    is_success,
                    ",".join(identifiers),
                    None,
                    codebleu_score,
                    None,
                    None,
                    None,
                    query_times,
                ]
            ]
        else:
            if flag_success == True:
                is_success = 1
            else:
                is_success = -1
            res = []
            for indi in self.parent_pop.indi:  # save each individual in the population
                if (
                    indi.obj_[0] < 0.5 * codebleu_score
                ):  # this individual is an adversarial example
                    success = 1
                else:
                    success = -1
                res.append(
                    [
                        idx,
                        source,
                        len(source),
                        " ".join(indi.tokens_),
                        indi.label_,
                        target,
                        success,
                        ",".join(identifiers),
                        ",".join(indi.identifiers_),
                        codebleu_score,
                        indi.obj_[0],
                        indi.obj_[1],
                        indi.obj_[2],
                        query_times,
                    ]
                )
        df = pd.DataFrame(data=res)
        df.to_csv(self.args.csv_store_path, mode="a+", header=False)
        logger.info("******End Attack, result={}******".format(is_success))
        return is_success
