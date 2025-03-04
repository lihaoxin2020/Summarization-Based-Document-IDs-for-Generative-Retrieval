import torch
import time
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, DistributedSampler
from transformers import (
    AutoTokenizer,
    Trainer,
    Seq2SeqTrainer,
    LogitsProcessorList,
    PrefixConstrainedLogitsProcessor,
    StoppingCriteriaList
)
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
    nested_concat,
    nested_numpify,
    nested_truncate,
    find_batch_size,
)
from transformers.trainer_utils import (
    denumpify_detensorize,
    EvalLoopOutput,
    EvalPrediction,
    has_length,
    seed_worker
)
from transformers.training_args import ParallelMode, TrainingArguments
from transformers.utils import is_torch_tpu_available
from transformers.trainer import logger
from transformers.deepspeed import is_deepspeed_zero3_enabled, deepspeed_init
from typing import Any, Callable, Dict, Union, Optional, List, Tuple

import numpy as np
import json
from .ConstraintDecoding import PrefixAllowedKeywords, RetrievalStoppingCriteria

# id_label = "key"
# with open("beir_msmarco/beir_msmarco-corpus-100k-ckpt.json") as f:
#     passages_keywords = [ex[id_label] for ex in json.load(f)]
# with open("beir_msmarco/beir_msmarco-corpus-100k-l30.json") as f:
#     passages_keywords = [ex[id_label] for ex in json.load(f)]
# with open("beir_msmarco/beir_msmarco-corpus-100k-clustering.json") as f:
#     passages_keywords = [ex[id_label] for ex in json.load(f)]
# with open("mm2-data/mm2-100k/mm2-100k-test-d2k.json") as f:
#     test_keywords = [value[id_label] for _, value in json.load(f).items()]
# passages_keywords = train_keywords + valid_keywords + test_keywords
# with open("beir_msmarco/beir_msmarco-corpus-100k-semantic_ids.json") as f:
#     passages_keywords = list(json.load(f).values())


# tokenizer = AutoTokenizer.from_pretrained(
#     "EleutherAI/pythia-160m-deduped",
#     cache_dir="../.cache",
#     use_fast=True
# )

# tokenizer = AutoTokenizer.from_pretrained(
#     "google/t5-v1_1-base",
#     cache_dir="../.cache",
#     use_fast=True
# )
# logits_processor = LogitsProcessorList([
#     PrefixConstrainedLogitsProcessor(
#         PrefixAllowedKeywords(
#             tokenizer("\nKeys:").input_ids,
#             # [tokenizer.pad_token_id],
#             [tokenizer(" " + k.strip()).input_ids for k in passages_keywords],
#                                                tokenizer.eos_token_id
#         ),
#         num_beams=10
#     ),
# ])

# stopping_criteria = RetrievalStoppingCriteria(
#     id_trie=logits_processor[0]._prefix_allowed_tokens_fn.trie, 
#     keyword_start_ids=tokenizer("\nKeys:").input_ids,
#     stops=[tokenizer.eos_token_id]
# )
# del tokenizer


class RetrievalTrainer(Seq2SeqTrainer):
    def __init__(self, valid_id_list, keyword_start, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if valid_id_list is not None:
            self.logits_processor = LogitsProcessorList([
                PrefixConstrainedLogitsProcessor(
                    PrefixAllowedKeywords(
                        self.tokenizer(keyword_start).input_ids if isinstance(keyword_start, str) else keyword_start,
                        # [tokenizer.pad_token_id],
                        [self.tokenizer(" " + k.strip()).input_ids for k in valid_id_list],
                        self.tokenizer.eos_token_id
                    ),
                    num_beams=self.args.generation_num_beams
                ),
            ])

            self.stopping_criteria = RetrievalStoppingCriteria(
                id_trie=self.logits_processor[0]._prefix_allowed_tokens_fn.trie, 
                keyword_start_ids=self.tokenizer(keyword_start).input_ids if isinstance(keyword_start, str) else keyword_start,
                stops=[self.tokenizer.eos_token_id]
            )
        else:
            self.logits_processor = None
            self.stopping_criteria = None
        

    # Copied from Accelerate.
    def _pad_across_processes(self, tensor, pad_index=-100):
        """
        Recursively pad the tensors in a nested list/tuple/dictionary of tensors from all devices to the same size so
        they can safely be gathered.
        """
        if isinstance(tensor, (list, tuple)):
            return type(tensor)(self._pad_across_processes(t, pad_index=pad_index) for t in tensor)
        elif isinstance(tensor, dict):
            return type(tensor)({k: self._pad_across_processes(v, pad_index=pad_index) for k, v in tensor.items()})
        elif not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"Can't pad the values of type {type(tensor)}, only of nested list/tuple/dicts of tensors."
            )

        if len(tensor.shape) < 2:
            return tensor
        # Gather all sizes
        size = torch.tensor(tensor.shape, device=tensor.device)[None]
        sizes = self._nested_gather(size).cpu()

        max_size = max(s[1] for s in sizes)
        # When extracting XLA graphs for compilation, max_size is 0,
        # so use inequality to avoid errors.
        if tensor.shape[1] >= max_size:
            return tensor

        # Then pad to the maximum size
        old_size = tensor.shape
        new_size = list(old_size)
        new_size[1] = max_size
        new_tensor = tensor.new_zeros(tuple(new_size)) + pad_index
        new_tensor[:, : old_size[1]] = tensor
        return new_tensor
    

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:
            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            inputs_decode = self._prepare_input(inputs["input_ids"]) if args.include_inputs_for_metrics else None

            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
            if inputs_decode is not None:
                inputs_decode = self._pad_across_processes(inputs_decode)
                inputs_decode = self._nested_gather(inputs_decode)
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self._pad_across_processes(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self._nested_gather(logits)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            if labels is not None:
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        # if all_losses is not None:
        #     all_losses = all_losses[:num_samples]
        # if all_preds is not None:
        #     all_preds = nested_truncate(all_preds, num_samples)
        # if all_labels is not None:
        #     all_labels = nested_truncate(all_labels, num_samples)
        # if all_inputs is not None:
        #     all_inputs = nested_truncate(all_inputs, num_samples)

        # Metrics!
        # if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
        if self.compute_metrics is not None:
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
                )
            else:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    # def evaluation_loop(
    #     self,
    #     dataloader: DataLoader,
    #     description: str,
    #     prediction_loss_only: Optional[bool] = None,
    #     ignore_keys: Optional[List[str]] = None,
    #     metric_key_prefix: str = "eval",
    # ) -> EvalLoopOutput:
    #     """
    #     Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

    #     Works both with or without labels.
    #     """
    #     args = self.args

    #     prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

    #     # if eval is called w/o train, handle model prep here
    #     if self.is_deepspeed_enabled and self.deepspeed is None:
    #         _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

    #     model = self._wrap_model(self.model, training=False, dataloader=dataloader)

    #     if len(self.accelerator._models) == 0 and model is self.model:
    #         model = (
    #             self.accelerator.prepare(model)
    #             if self.is_deepspeed_enabled
    #             else self.accelerator.prepare_model(model, evaluation_mode=True)
    #         )

    #         if self.is_fsdp_enabled:
    #             self.model = model

    #         # for the rest of this function `model` is the outside model, whether it was wrapped or not
    #         if model is not self.model:
    #             self.model_wrapped = model

    #         # backward compatibility
    #         if self.is_deepspeed_enabled:
    #             self.deepspeed = self.model_wrapped

    #     # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
    #     # while ``train`` is running, cast it to the right dtype first and then put on device
    #     if not self.is_in_train:
    #         if args.fp16_full_eval:
    #             model = model.to(dtype=torch.float16, device=args.device)
    #         elif args.bf16_full_eval:
    #             model = model.to(dtype=torch.bfloat16, device=args.device)

    #     batch_size = self.args.eval_batch_size

    #     logger.info(f"***** Running {description} *****")
    #     if has_length(dataloader):
    #         logger.info(f"  Num examples = {self.num_examples(dataloader)}")
    #     else:
    #         logger.info("  Num examples: Unknown")
    #     logger.info(f"  Batch size = {batch_size}")

    #     model.eval()

    #     self.callback_handler.eval_dataloader = dataloader
    #     # Do this before wrapping.
    #     eval_dataset = getattr(dataloader, "dataset", None)

    #     if args.past_index >= 0:
    #         self._past = None

    #     # Initialize containers
    #     # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
    #     losses_host = None
    #     preds_host = None
    #     labels_host = None
    #     inputs_host = None

    #     # losses/preds/labels on CPU (final containers)
    #     all_losses = None
    #     all_preds = None
    #     all_labels = None
    #     all_inputs = None
    #     # Will be useful when we have an iterable dataset so don't know its length.

    #     observed_num_examples = 0
    #     # Main evaluation loop
    #     for step, inputs in enumerate(dataloader):
    #         # Update the observed num examples
    #         observed_batch_size = find_batch_size(inputs)
    #         if observed_batch_size is not None:
    #             observed_num_examples += observed_batch_size
    #             # For batch samplers, batch_size is not known by the dataloader in advance.
    #             if batch_size is None:
    #                 batch_size = observed_batch_size

    #         # Prediction step
    #         loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
    #         main_input_name = getattr(self.model, "main_input_name", "input_ids")
    #         inputs_decode = self._prepare_input(inputs[main_input_name]) if args.include_inputs_for_metrics else None

    #         if is_torch_tpu_available():
    #             xm.mark_step()

    #         # Update containers on host
    #         if loss is not None:
    #             losses = self.accelerator.gather_for_metrics((loss.repeat(batch_size)))
    #             losses_host = losses if losses_host is None else nested_concat(losses_host, losses, padding_index=-100)
    #         if labels is not None:
    #             labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
    #         if inputs_decode is not None:
    #             inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
    #             inputs_decode = self.accelerator.gather_for_metrics((inputs_decode))
    #             inputs_host = (
    #                 inputs_decode
    #                 if inputs_host is None
    #                 else nested_concat(inputs_host, inputs_decode, padding_index=-100)
    #             )
    #         if logits is not None:
    #             logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
    #             if self.preprocess_logits_for_metrics is not None:
    #                 logits = self.preprocess_logits_for_metrics(logits, labels)
    #             # logits = self.accelerator.gather_for_metrics((logits))
    #             preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)

    #         if labels is not None:
    #             labels = self.accelerator.gather_for_metrics((labels))
    #             labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)

    #         self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

    #         # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
    #         if (
    #             args.eval_accumulation_steps is not None
    #             and (step + 1) % args.eval_accumulation_steps == 0
    #             and self.accelerator.sync_gradients
    #         ):
    #             if losses_host is not None:
    #                 losses = nested_numpify(losses_host)
    #                 all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
    #             if preds_host is not None:
    #                 logits = nested_numpify(preds_host)
    #                 all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
    #             if inputs_host is not None:
    #                 inputs_decode = nested_numpify(inputs_host)
    #                 all_inputs = (
    #                     inputs_decode
    #                     if all_inputs is None
    #                     else nested_concat(all_inputs, inputs_decode, padding_index=-100)
    #                 )
    #             if labels_host is not None:
    #                 labels = nested_numpify(labels_host)
    #                 all_labels = (
    #                     labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
    #                 )

    #             # Set back to None to begin a new accumulation
    #             losses_host, preds_host, inputs_host, labels_host = None, None, None, None

    #     if args.past_index and hasattr(self, "_past"):
    #         # Clean the state at the end of the evaluation loop
    #         delattr(self, "_past")

    #     # Gather all remaining tensors and put them back on the CPU
    #     if losses_host is not None:
    #         losses = nested_numpify(losses_host)
    #         all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
    #     if preds_host is not None:
    #         logits = nested_numpify(preds_host)
    #         all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
    #     if inputs_host is not None:
    #         inputs_decode = nested_numpify(inputs_host)
    #         all_inputs = (
    #             inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
    #         )
    #     if labels_host is not None:
    #         labels = nested_numpify(labels_host)
    #         all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

    #     # Number of samples
    #     if has_length(eval_dataset):
    #         num_samples = len(eval_dataset)
    #     # The instance check is weird and does not actually check for the type, but whether the dataset has the right
    #     # methods. Therefore we need to make sure it also has the attribute.
    #     elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
    #         num_samples = eval_dataset.num_examples
    #     else:
    #         if has_length(dataloader):
    #             num_samples = self.num_examples(dataloader)
    #         else:  # both len(dataloader.dataset) and len(dataloader) fail
    #             num_samples = observed_num_examples
    #     if num_samples == 0 and observed_num_examples > 0:
    #         num_samples = observed_num_examples

    #     # Metrics!
    #     # if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
    #     if self.compute_metrics is not None and all_preds is not None:
    #         if args.include_inputs_for_metrics:
    #             metrics = self.compute_metrics(
    #                 EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
    #             )
    #         else:
    #             metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
    #     else:
    #         metrics = {}

    #     # To be JSON-serializable, we need to remove numpy types or zero-d tensors
    #     metrics = denumpify_detensorize(metrics)

    #     if all_losses is not None:
    #         metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
    #     if hasattr(self, "jit_compilation_time"):
    #         metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

    #     # Prefix all keys with metric_key_prefix + '_'
    #     for key in list(metrics.keys()):
    #         if not key.startswith(f"{metric_key_prefix}_"):
    #             metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

    #     return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        # Priority (handled in generate):
        # gen_kwargs > model.generation_config > default GenerationConfig()
        gen_kwargs = self._gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.model.config.max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.model.config.num_beams
        )
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )

        #######
        gen_kwargs["num_return_sequences"] = gen_kwargs["num_beams"]
        gen_kwargs["eos_token_id"] = 0
        gen_kwargs["pad_token_id"] = 1
        # gen_kwargs["max_new_tokens"] = 512
        gen_kwargs["early_stopping"] = True
        # gen_kwargs.pop("max_length")

        # logits_processor = LogitsProcessorList([
        #     PrefixConstrainedLogitsProcessor(
        #         PrefixAllowedKeywords(
        #             self.tokenizer("\nKeys:").input_ids,
        #             [self.tokenizer(" " + k.strip()).input_ids for k in passages_keywords],
        #                                                self.tokenizer.eos_token_id
        #         ),
        #         num_beams=gen_kwargs["num_beams"]
        #     ),
        # ])

        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if (
            "labels" in inputs
            and "decoder_input_ids" in inputs
            and inputs["labels"].shape == inputs["decoder_input_ids"].shape
        ):
            inputs = {k: v for k, v in inputs.items() if k != "decoder_input_ids"}
        # stopping_criteria = RetrievalStoppingCriteria(id_trie=logits_processor[0]._prefix_allowed_tokens_fn.trie, stops=[0])
        # print("****start generation****")
        # start = time.time()
        generated_tokens = self.model.generate(
            **inputs,
            logits_processor=self.logits_processor if self.logits_processor is not None else None,
            stopping_criteria=StoppingCriteriaList([self.stopping_criteria]) if self.stopping_criteria is not None else None,
            **gen_kwargs
        )
        # end = time.time()
        # print(f"****finishing after {end - start}s****")
        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        # TODO: remove this hack when the legacy code that initializes generation_config from a model config is
        # removed in https://github.com/huggingface/transformers/blob/98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
        elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)

        # with torch.no_grad():
            # if has_labels:
            #     with self.compute_loss_context_manager():
            #         outputs = model(**inputs)
            #     if self.label_smoother is not None:
            #         loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
            #     else:
            #         loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            # else:
        # Evaluated by retireval recall, not loss. 
        loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_config.max_length:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
            elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
        else:
            labels = None

        return loss, generated_tokens, labels

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        # if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
        #     train_dataset = self._remove_unused_columns(train_dataset, description="training")
        # else:
        #     data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self._train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return DataLoader(
                train_dataset,
                batch_size=self._train_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
        )

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        generator = None
        if self.args.world_size <= 1:
            generator = torch.Generator()
            # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
            # `args.seed`) if data_seed isn't provided.
            # Further on in this method, we default to `args.seed` instead.
            if self.args.data_seed is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = self.args.data_seed
            generator.manual_seed(seed)

        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            if self.args.world_size <= 1:
                return LengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    dataset=self.train_dataset,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    generator=generator,
                )
            else:
                return DistributedLengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    dataset=self.train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    seed=seed,
                )

        else:
            if self.args.world_size <= 1:
                return RandomSampler(self.train_dataset, generator=generator)
            elif (
                self.args.parallel_mode in [ParallelMode.TPU, ParallelMode.SAGEMAKER_MODEL_PARALLEL]
                and not self.args.dataloader_drop_last
            ):
                # Use a loop for TPUs when drop_last is False to have all batches have the same size.
                return DistributedSamplerWithLoop(
                    self.train_dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=seed,
                )
            else:
                return DistributedSampler(
                    self.train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=seed,
                    # shuffle=False,
                )
