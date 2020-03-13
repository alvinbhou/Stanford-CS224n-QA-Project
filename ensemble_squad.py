# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""


from args import get_bert_args
import glob
# import logging
import os
import random
import timeit
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, ConcatDataset, TensorDataset, ChainDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import util
import collections
import pickle
import hack
from itertools import product
from models.ensemble import EnsembleQA, EnsembleStackQA

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForQuestionAnswering,
    AlbertTokenizer,
    BertConfig,
    BertForQuestionAnswering,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForQuestionAnswering,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForQuestionAnswering,
    RobertaTokenizer,
    XLMConfig,
    XLMForQuestionAnswering,
    XLMTokenizer,
    XLNetConfig,
    XLNetForQuestionAnswering,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

# Use logger from util.py instead
# logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (BertConfig, RobertaConfig, XLNetConfig, XLMConfig)
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForQuestionAnswering, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForQuestionAnswering, RobertaTokenizer),
    "xlnet": (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(args.save_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    # Added here for reproductibility
    set_seed(args)

    # global var for current best f1
    cur_best_f1 = 0.0

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "predict_start_logits": batch[0],
                "predict_end_logits": batch[1],
                "start_positions": batch[2],
                "end_positions": batch[3],
            }

            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]
            # print(f'loss at step: {global_step} loss {loss}')
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
                # logger.info(f"Weights at: {model.weights}")

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                eval_results = None  # Evaluation result

                # Log metrics
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.local_rank == -1 and args.evaluate_during_training:
                        # Create output dir/path
                        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        output_path = os.path.join(output_dir, 'eval_result.json')

                        # Get eval results and save the log to output path
                        eval_results, all_predictions = evaluate(args, model, tokenizer, save_dir=output_dir, save_log_path=output_path)
                        for key, value in eval_results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)

                        util.save_json_file(os.path.join(output_dir, 'predictions_.json'), all_predictions)
                        util.convert_submission_format_and_save(
                            output_dir, prediction_file_path=os.path.join(
                                output_dir, 'predictions_.json'))

                        # log eval result
                        logger.info(f"Evaluation result at {global_step} step: {eval_results}")
                        # logger.info(f"Weights at {model.weights}")
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                # Save model checkpoint
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    output_dir = os.path.join(
                        args.output_dir, 'cur_best') if args.save_best_only else os.path.join(
                        args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    # Get eval results and save the log to output path
                    if args.local_rank in [-1, 0] and args.evaluate_during_saving:
                        output_path = os.path.join(output_dir, 'eval_result.json')
                        eval_results, all_predictions = evaluate(args, model, tokenizer, save_dir=output_dir, save_log_path=None)
                        for key, value in eval_results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                        # log eval result
                        logger.info(f"Evaluation result at {global_step} step: {eval_results}")
                        # logger.info(f"Weights at {model.weights}")
                        # save current result at args.output_dir
                        if os.path.exists(os.path.join(args.output_dir, "eval_result.json")):
                            util.read_and_update_json_file(os.path.join(args.output_dir, "eval_result.json"), {global_step: eval_results})
                        else:
                            util.save_json_file(os.path.join(args.output_dir, "eval_result.json"), {global_step: eval_results})

                     # Save cur best model only
                    # Take care of distributed/parallel training
                    if (eval_results and cur_best_f1 < eval_results['f1']) or not args.save_best_only:
                        if eval_results and cur_best_f1 < eval_results['f1']:
                            cur_best_f1 = eval_results['f1']
                        model_to_save = model.module if hasattr(model, "module") else model
                        # model_to_save.save_pretrained(output_dir)  # BertQA is not a PreTrainedModel class

                        torch.save(model_to_save, os.path.join(output_dir, 'pytorch_model.bin'))  # save entire model
                        tokenizer.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)

                        util.save_json_file(os.path.join(output_dir, 'predictions_.json'), all_predictions)
                        util.convert_submission_format_and_save(
                            output_dir, prediction_file_path=os.path.join(
                                output_dir, 'predictions_.json'))
                        if args.save_best_only:
                            key = ','.join(map(str, model.weights.tolist())) if args.do_weighted_ensemble else global_step
                            util.save_json_file(os.path.join(output_dir, "eval_result.json"), {key: eval_results})

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix="", save_dir='', save_log_path=None, return_predicts=True):
    examples, features, dataset, tokenizer, n_models = load_combined_examples(args, evaluate=True)

    if not save_dir and args.local_rank in [-1, 0]:
        os.makedirs(save_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "predict_start_logits": batch[0],
                "predict_end_logits": batch[1],
            }
            example_indices = batch[2]
            outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            is_impossible = eval_feature.is_impossible

            output = [to_list(output[i]) for output in outputs]
            start_logits, end_logits = output
            result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    # Compute predictions
    output_prediction_file = os.path.join(save_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(save_dir, "nbest_predictions_{}.json".format(prefix))

    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(save_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    predictions, probs = hack.compute_predictions_logits(
        examples,
        features,
        all_results,
        args.n_best_size,
        args.max_answer_length,
        args.do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        args.verbose_logging,
        args.version_2_with_negative,
        args.null_score_diff_threshold,
        tokenizer,
        prob_mode='add'
    )

    # Compute the F1 and exact scores.
    results = squad_evaluate(examples, predictions)

    # save log to file
    if save_log_path:
        util.save_json_file(save_log_path, results)

    if return_predicts:
        return results, predictions
    return results


def ensemble_vote(args, save_dir='', save_log_path=None, prefix='', predict_prob_mode='add'):
    examples, all_model_features, all_model_results, tokenizers = load_saved_examples(args, evaluate=True)

    if not save_dir and args.local_rank in [-1, 0]:
        os.makedirs(save_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    # eval_sampler = SequentialSampler(dataset)
    # eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info(f"***** Running ensemble {prefix}*****")
    logger.info("  Num examples = %d", len(examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    # We do pure voting now, not taking new inputs
    # start_time = timeit.default_timer()
    # evalTime = timeit.default_timer() - start_time
    # logger.info(" Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    # Compute predictions
    output_prediction_file = os.path.join(save_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(save_dir, "nbest_predictions_{}.json".format(prefix))
    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(save_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    all_predictions = []
    all_probs = []
    logger.info(f'predict_prob_mode: {predict_prob_mode}')
    for model_idx in tqdm(range(len(tokenizers)), desc="Predicting"):
        features = all_model_features[model_idx]
        all_results = all_model_results[model_idx]
        tokenizer = tokenizers[model_idx]

        predictions, probs = hack.compute_predictions_logits(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            args.do_lower_case,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            args.verbose_logging,
            args.version_2_with_negative,
            args.null_score_diff_threshold,
            tokenizer,
            prob_mode=predict_prob_mode
        )
        all_predictions.append(predictions)
        all_probs.append(probs)
        # continue

    # num of predictions
    num_of_predicions = len(all_predictions[0])
    logger.info(f'Number of predicions {num_of_predicions}')

    final_predictions = collections.OrderedDict()
    output_result = collections.OrderedDict()
    # Grid Search
    if args.do_grid_search:
        grid_search_results = collections.OrderedDict()
        grid_search_predictions = collections.OrderedDict()
        for weights in product(np.arange(6), repeat=len(all_probs)):
            if weights == (0, 0, 0, 0, 0):
                continue
            for qas_id in all_predictions[0].keys():
                probs = np.array([d_prob[qas_id] for d_prob in all_probs])
                for i, w in enumerate(weights):
                    probs[i] *= w

                idx = np.argmax(probs)
                final_predictions[qas_id] = all_predictions[idx][qas_id]

            """
            logger.info('Model individual results')
            for i in range(len(tokenizers)):
                results = squad_evaluate(examples, all_predictions[i])
                logger.info(results)
            """
            # Compute the F1 and exact scores.
            logger.info(f'Weights: {weights}')
            logger.info('Ensemble results')
            final_results = squad_evaluate(examples, final_predictions)
            logger.info(final_results)

            if len(grid_search_results) == 0:
                best_weights = weights
                grid_search_results = final_results
                grid_search_predictions = final_predictions
            else:
                if grid_search_results['exact'] + grid_search_results['f1'] < final_results['exact'] + final_results['f1']:
                    best_weights = weights
                    grid_search_results = final_results
                    grid_search_predictions = final_predictions
        # save log to file
        logger.info(f'Best Weights: {best_weights}')
        output_result[best_weights] = grid_search_results
        util.save_json_file(os.path.join(save_dir, 'eval_results.json'), output_result)

        # save prediction to file
        # TODO save grid search best
        util.save_json_file(os.path.join(save_dir, 'predictions_.json'), grid_search_predictions)
        util.convert_submission_format_and_save(
            save_dir, prediction_file_path=os.path.join(
                save_dir, 'predictions_.json'))

        return grid_search_results
    else:
        for qas_id in all_predictions[0].keys():
            probs = np.array([d_prob[qas_id] for d_prob in all_probs])

            idx = np.argmax(probs)
            final_predictions[qas_id] = all_predictions[idx][qas_id]

        logger.info('Model individual results')
        for i in range(len(tokenizers)):
            results = squad_evaluate(examples, all_predictions[i])
            logger.info(results)
        # Compute the F1 and exact scores.
        logger.info('Ensemble results')
        final_results = squad_evaluate(examples, final_predictions)
        logger.info(final_results)

        # save log to file
        util.save_json_file(os.path.join(save_dir, 'eval_results.json'), final_results)

        util.save_json_file(os.path.join(save_dir, 'predictions_.json'), final_predictions)
        util.convert_submission_format_and_save(
            save_dir, prediction_file_path=os.path.join(
                save_dir, 'predictions_.json'))

        return final_results


def load_saved_examples(args, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    if not args.data_dir and ((evaluate and not args.predict_file) or (not evaluate and not args.train_file)):
        try:
            import tensorflow_datasets as tfds
        except ImportError:
            raise ImportError("If not data_dir is specified, tensorflow_datasets needs to be installed.")

        if args.version_2_with_negative:
            logger.warn("tensorflow_datasets does not handle version 2 of SQuAD.")
        logger.warn("Something went wrong!")
        tfds_examples = tfds.load("squad")
        examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate)
    else:
        processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
        if evaluate:
            examples = processor.get_dev_examples(args.data_dir, filename=args.predict_file)
            # Sanity check for loading the correct example
            assert examples[0].question_text == 'In what country is Normandy located?', 'Invalid dev file!'
        else:
            # Normal get train examples
            examples = processor.get_train_examples(args.data_dir, filename=args.train_file)
            # Sanity check for loading the correct example
            assert examples[0].question_text == 'When did Beyonce start becoming popular?', 'Invalid train file!'
    assert args.saved_processed_data_dir, 'args.saved_processed_data_dir not defined!'
    ensemble_dir = args.saved_processed_data_dir

    print(args.saved_processed_data_dir)
    if evaluate:
        with open(os.path.join(ensemble_dir, 'saved_data_dev.pkl'), 'rb') as f:
            saved_data = pickle.load(f)
    else:
        with open(os.path.join(ensemble_dir, 'saved_data_train.pkl'), 'rb') as f:
            saved_data = pickle.load(f)
    # saved_data: [features, all_results, tokenizer]
    features, all_results, tokenizer = saved_data

    if evaluate:
        assert len(examples) == 6078
    else:
        assert len(examples) == 130319

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()
    return examples, features, all_results, tokenizer


def load_combined_examples(args, evaluate=False):
    """
    Deprecated sadly
    """
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    if not args.data_dir and ((evaluate and not args.predict_file) or (not evaluate and not args.train_file)):
        try:
            import tensorflow_datasets as tfds
        except ImportError:
            raise ImportError("If not data_dir is specified, tensorflow_datasets needs to be installed.")

        if args.version_2_with_negative:
            logger.warn("tensorflow_datasets does not handle version 2 of SQuAD.")
        logger.warn("Something went wrong!")
        tfds_examples = tfds.load("squad")
        examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate)
    else:
        processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
        if evaluate:
            examples = processor.get_dev_examples(args.data_dir, filename=args.predict_file)
            # Sanity check for loading the correct example
            assert examples[0].question_text == 'In what country is Normandy located?', 'Invalid dev file!'
        else:
            # Normal get train examples
            examples = processor.get_train_examples(args.data_dir, filename=args.train_file)
            # Sanity check for loading the correct example
            assert examples[0].question_text == 'When did Beyonce start becoming popular?', 'Invalid train file!'

    assert args.saved_processed_data_dir, 'args.saved_processed_data_dir not defined!'
    ensemble_dir = args.saved_processed_data_dir

    if evaluate:
        with open(os.path.join(ensemble_dir, 'saved_data_dev.pkl'), 'rb') as f:
            saved_data = pickle.load(f)
    else:
        with open(os.path.join(ensemble_dir, 'saved_data_train.pkl'), 'rb') as f:
            saved_data = pickle.load(f)
    # saved_data: [features, all_results, tokenizer]
    features, combined_all_results, tokenizer = saved_data
    assert np.array_equal([f.start_position for f in features[0]], [f.start_position for f in features[1]]), print("Same family Same features")

    # Same family same feature and tokenizer, so we pick the first one
    features = features[0]
    tokenizer = tokenizer[0]
    all_predict_start_logits = []
    all_predict_end_logits = []
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)

    for all_results in combined_all_results:
        all_predict_start_logits.append(torch.tensor([s.start_logits for s in all_results], dtype=torch.float))
        all_predict_end_logits.append(torch.tensor([s.end_logits for s in all_results], dtype=torch.float))

    if evaluate:
        all_example_indices = torch.arange(all_input_ids.size(0), dtype=torch.long)

    all_predict_start_logits = torch.stack(all_predict_start_logits).permute(1, 0, 2)
    all_predict_end_logits = torch.stack(all_predict_end_logits).permute(1, 0, 2)

    # print(f'all_input_ids: {all_input_ids.shape}, all_predict_start_logits{all_predict_start_logits.shape}, all_predict_end_logits:{all_predict_end_logits.shape}')

    if evaluate:
        dataset = TensorDataset(
            all_predict_start_logits, all_predict_end_logits, all_example_indices
        )
    else:
        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
        # print(all_start_positions.shape, all_end_positions.shape)
        dataset = TensorDataset(
            all_predict_start_logits, all_predict_end_logits, all_start_positions, all_end_positions
        )
    if evaluate:
        assert len(examples) == 6078
    else:
        assert len(examples) == 130319

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()
    return examples, features, dataset, tokenizer, len(combined_all_results)


def main():
    args = get_bert_args()

    assert not (args.do_output and args.do_train), 'Don\'t output and train at the same time!'
    if args.do_output:
        sub_dir_prefix = 'output'
    elif args.do_train:
        sub_dir_prefix = 'train'
    else:
        sub_dir_prefix = 'test'
    # No matter what, we do ensemble here lol
    sub_dir_prefix = 'ensemble3'
    args.save_dir = util.get_save_dir(args.save_dir, args.name, sub_dir_prefix)
    args.output_dir = args.save_dir

    global logger
    logger = util.get_logger(args.save_dir, args.name)

    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )

    if not args.evaluate_during_saving and args.save_best_only:
        raise ValueError("No best result without evaluation during saving")

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # Training
    if args.do_train and (args.do_weighted_ensemble or args.do_stack_ensemble):
        examples, features, train_dataset, tokenizer, n_models = load_combined_examples(args, evaluate=False)
        model = EnsembleQA(n_models) if args.do_weighted_ensemble else EnsembleStackQA(n_models)
        model = model.to(args.device)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and the tokenizer
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        model_to_save = model.module if hasattr(model, "module") else model
        # model_to_save.save_pretrained(output_dir)  # BertQA is not a PreTrainedModel class
        torch.save(model_to_save, os.path.join(args.output_dir, 'pytorch_model.bin'))  # save entire model
        tokenizer.save_pretrained(args.output_dir)  # save tokenizer

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = torch.load(os.path.join(args.output_dir, 'cur_best', 'pytorch_model.bin'))
        model.to(args.device)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        if args.do_train:
            logger.info("Loading checkpoints saved during training for evaluation")
            checkpoints = [args.output_dir]
            if args.eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c)
                    for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
                )
                # logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs
        else:
            logger.info("Loading checkpoint %s for evaluation", args.model_name_or_path)
            checkpoints = [args.eval_dir]
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            # Load a trained model and vocabulary that you have fine-tuned
            model = torch.load(os.path.join(args.output_dir, 'cur_best', 'pytorch_model.bin'))
            model.to(args.device)

            # Evaluate
            result, all_predictions = evaluate(
                args,
                model,
                tokenizer,
                prefix=global_step,
                save_dir=args.output_dir,
                save_log_path=os.path.join(
                    checkpoint,
                    'eval_result.json'))

            result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
            results.update(result)

            logger.info(f'Convert format and Writing submission file to directory {args.output_dir}...')
            util.save_json_file(os.path.join(args.output_dir, 'cur_best', 'predictions_.json'), all_predictions)
            util.convert_submission_format_and_save(
                args.output_dir, prediction_file_path=os.path.join(
                    args.output_dir, 'cur_best', 'predictions_.json'))

    logger.info("Results: {}".format(results))

    # Generate ensemble output
    if args.do_ensemble_voting and args.local_rank in [-1, 0]:
        results = ensemble_vote(args, save_dir=args.save_dir, predict_prob_mode='add')

    return results
    # load_combined_examples(args, evaluate=True)


if __name__ == "__main__":
    main()
