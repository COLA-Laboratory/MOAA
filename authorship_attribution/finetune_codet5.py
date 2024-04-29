import os
import logging
import argparse
import math
import numpy as np
from io import open
from tqdm import tqdm
import random
import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    SequentialSampler,
    RandomSampler,
    TensorDataset,
)
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    get_linear_schedule_with_warmup,
    RobertaConfig,
    RobertaModel,
    RobertaTokenizer,
    BartConfig,
    BartForConditionalGeneration,
    BartTokenizer,
    T5Config,
    T5ForConditionalGeneration,
    T5Tokenizer,
)
import multiprocessing
import time

from model import CodeT5Model
from model import get_model_size
from _utils import load_and_cache_data

MODEL_CLASSES = {
    "roberta": (RobertaConfig, RobertaModel, RobertaTokenizer),
    "t5": (T5Config, T5ForConditionalGeneration, T5Tokenizer),
    "codet5": (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
    "bart": (BartConfig, BartForConditionalGeneration, BartTokenizer),
}

cpu_cont = multiprocessing.cpu_count()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def evaluate(args, model, eval_examples, eval_data, write_to_pred=False):
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(
        eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size
    )

    # Eval
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Num batches = %d", len(eval_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Evaluating"):
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        with torch.no_grad():
            lm_loss, logit = model(inputs, label)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    # preds = logits[:, 1] > 0.5
    preds = []
    for logit in logits:
        preds.append(np.argmax(logit))
    print(preds)
    eval_acc = np.mean(labels == preds)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)

    result = {
        "eval_loss": float(perplexity),
        "eval_acc": round(eval_acc, 4),
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    if write_to_pred:
        with open(os.path.join(args.output_dir, "predictions.txt"), "w") as f:
            for example, pred in zip(eval_examples, preds):
                if pred:
                    f.write(str(example.idx) + "\t1\n")
                else:
                    f.write(str(example.idx) + "\t0\n")

    return result


def main():
    parser = argparse.ArgumentParser()
    t0 = time.time()
    parser.add_argument(
        "--model_type",
        default="codet5",
        type=str,
        choices=["roberta", "bart", "codet5"],
    )
    parser.add_argument("--epoch", default=100, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--patience", default=5, type=int)
    parser.add_argument("--res_fn", type=str, default="")

    # Required parameters
    parser.add_argument(
        "--model_name_or_path",
        default="roberta-base",
        type=str,
        help="Path to pre-trained model: e.g. roberta-base",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--load_model_path",
        default=None,
        type=str,
        help="Path to trained model: Should contain the .bin files",
    )
    parser.add_argument("--cache_path", type=str, required=True)
    parser.add_argument("--data_num", default=-1, type=int)
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="roberta-base",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--max_source_length",
        default=64,
        type=int,
        help="The maximum total source sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_target_length",
        default=32,
        type=int,
        help="The maximum total target sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run eval on the train set."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_test", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )

    # Other parameters
    parser.add_argument(
        "--train_filename",
        default=None,
        type=str,
        help="The train filename. Should contain the .jsonl files for this task.",
    )
    parser.add_argument(
        "--dev_filename",
        default=None,
        type=str,
        help="The dev filename. Should contain the .jsonl files for this task.",
    )
    parser.add_argument(
        "--test_filename",
        default=None,
        type=str,
        help="The test filename. Should contain the .jsonl files for this task.",
    )

    parser.add_argument(
        "--train_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--beam_size", default=10, type=int, help="beam size for beam search"
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight deay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )

    parser.add_argument(
        "--save_steps",
        default=-1,
        type=int,
    )
    parser.add_argument(
        "--log_steps",
        default=-1,
        type=int,
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override epoch.",
    )
    parser.add_argument("--eval_steps", default=-1, type=int, help="")
    parser.add_argument("--train_steps", default=-1, type=int, help="")
    parser.add_argument(
        "--warmup_steps", default=100, type=int, help="Linear warmup over warmup_steps."
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="random seed for initialization"
    )
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, cpu count: %d",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        cpu_cont,
    )
    args.device = device

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # build model
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path
    )
    config.num_labels = 66
    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)

    model = CodeT5Model(model, config, tokenizer, args)
    logger.info(
        "Finish loading model [%s] from %s",
        get_model_size(model),
        args.model_name_or_path,
    )

    if args.load_model_path is not None:
        logger.info("Reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))

    model.to(device)

    pool = multiprocessing.Pool(cpu_cont)
    fa = open(os.path.join(args.output_dir, "summary.log"), "a+")

    # Train
    if args.do_train:
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Prepare training data loader
        train_example, train_data = load_and_cache_data(
            args, args.train_filename, pool, tokenizer, "train", is_sample=False
        )
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=args.train_batch_size
        )

        num_train_optimization_steps = args.epoch * len(train_dataloader)
        save_steps = max(len(train_dataloader), 1)
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
        )

        if args.warmup_steps < 1:
            warmup_steps = num_train_optimization_steps * args.warmup_steps
        else:
            warmup_steps = int(args.warmup_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_train_optimization_steps,
        )

        # Start training
        train_example_num = len(train_data)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info(
            "  Batch num = %d", math.ceil(train_example_num / args.train_batch_size)
        )
        logger.info("  Num epoch = %d", args.epoch)

        global_step, best_acc = 0, 0
        not_acc_inc_cnt = 0
        is_early_stop = False
        for cur_epoch in range(args.start_epoch, int(args.epoch)):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            model.train()
            for step, batch in enumerate(bar):
                batch = tuple(t.to(device) for t in batch)
                source_ids, labels = batch

                loss, logits = model(source_ids, labels)

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()

                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if nb_tr_steps % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    train_loss = round(
                        tr_loss * args.gradient_accumulation_steps / nb_tr_steps, 4
                    )
                    bar.set_description(
                        "[{}] Train loss {}".format(cur_epoch, round(train_loss, 3))
                    )

                if (step + 1) % save_steps == 0 and args.do_eval:
                    logger.info("***** CUDA.empty_cache() *****")
                    torch.cuda.empty_cache()

                    eval_examples, eval_data = load_and_cache_data(
                        args,
                        args.dev_filename,
                        pool,
                        tokenizer,
                        "valid",
                        is_sample=False,
                    )

                    result = evaluate(args, model, eval_examples, eval_data)
                    eval_acc = result["eval_acc"]

                    # save last checkpoint
                    last_output_dir = os.path.join(args.output_dir, "checkpoint-last")
                    if not os.path.exists(last_output_dir):
                        os.makedirs(last_output_dir)

                    if True or args.data_num == -1 and args.save_last_checkpoints:
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )
                        output_model_file = os.path.join(
                            last_output_dir, "pytorch_model.bin"
                        )
                        torch.save(model_to_save.state_dict(), output_model_file)
                        logger.info("Save the last model into %s", output_model_file)

                    if eval_acc > best_acc:
                        not_acc_inc_cnt = 0
                        logger.info("  Best acc: %s", round(eval_acc, 4))
                        logger.info("  " + "*" * 20)
                        fa.write(
                            "[%d] Best acc changed into %.4f\n"
                            % (cur_epoch, round(eval_acc, 4))
                        )
                        best_acc = eval_acc
                        # Save best checkpoint for best ppl
                        output_dir = os.path.join(
                            args.output_dir, "checkpoint-best-acc"
                        )
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        if args.data_num == -1 or True:
                            model_to_save = (
                                model.module if hasattr(model, "module") else model
                            )
                            output_model_file = os.path.join(
                                output_dir, "pytorch_model.bin"
                            )
                            torch.save(model_to_save.state_dict(), output_model_file)
                            logger.info(
                                "Save the best ppl model into %s", output_model_file
                            )
                    else:
                        not_acc_inc_cnt += 1
                        logger.info(
                            "acc does not increase for %d epochs", not_acc_inc_cnt
                        )
                        if not_acc_inc_cnt > args.patience:
                            logger.info(
                                "Early stop as acc do not increase for %d times",
                                not_acc_inc_cnt,
                            )
                            fa.write(
                                "[%d] Early stop as not_acc_inc_cnt=%d\n"
                                % (cur_epoch, not_acc_inc_cnt)
                            )
                            is_early_stop = True
                            break

                model.train()
            if is_early_stop:
                break

            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()
    if args.do_test:
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", args.eval_batch_size)
        for criteria in ["best-acc"]:
            file = os.path.join(
                args.output_dir, "checkpoint-{}/pytorch_model.bin".format(criteria)
            )
            logger.info("Reload model from {}".format(file))
            model.load_state_dict(torch.load(file))

            if args.n_gpu > 1:
                # multi-gpu training
                model = torch.nn.DataParallel(model)
            eval_examples, eval_data = load_and_cache_data(
                args, args.test_filename, pool, tokenizer, "test", False
            )

            result = evaluate(args, model, eval_examples, eval_data, write_to_pred=True)
            logger.info("  test_acc=%.4f", result["eval_acc"])
            logger.info("  " + "*" * 20)

            fa.write("[%s] test-acc: %.4f\n" % (criteria, result["eval_acc"]))
            if args.res_fn:
                with open(args.res_fn, "a+") as f:
                    f.write("[Time: {}] {}\n".format(get_elapse_time(t0), file))
                    f.write("[%s] acc: %.4f\n\n" % (criteria, result["eval_acc"]))
    fa.close()


def get_elapse_time(t0):
    elapse_time = time.time() - t0
    if elapse_time > 3600:
        hour = int(elapse_time // 3600)
        minute = int((elapse_time % 3600) // 60)
        return "{}h{}m".format(hour, minute)
    else:
        minute = int((elapse_time % 3600) // 60)
        return "{}m".format(minute)


if __name__ == "__main__":
    main()
