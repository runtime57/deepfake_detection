from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
import torch

class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    # def process_batch(self, batch, metrics: MetricTracker, gradient_accum_steps: int = 16):
    #     # initialize gradient step counter if not present
    #     if not hasattr(self, "_grad_step"):
    #         self._grad_step = 0

    #     batch = self.move_batch_to_device(batch)
    #     batch = self.transform_batch(batch)

    #     metric_funcs = self.metrics["inference"]
    #     if self.is_train:
    #         metric_funcs = self.metrics["train"]

    #         # Only zero grad if starting a new accumulation cycle
    #         if self._grad_step % gradient_accum_steps == 0:
    #             self.optimizer.zero_grad()
    
    #     with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
    #         outputs = self.model(**batch)
    #         batch.update(outputs)

    #         all_losses = self.criterion(**batch)
    #         batch.update(all_losses)

    #     if self.is_train:
    #         # scale loss for gradient accumulation
    #         loss = batch["loss"] / gradient_accum_steps
    #         loss.backward()

    #         self._grad_step += 1  # update counter

    #         # perform optimizer update only every gradient_accum_steps steps
    #         if self._grad_step % gradient_accum_steps == 0:
    #             self._clip_grad_norm()
    #             self.optimizer.step()

    #             if self.lr_scheduler is not None:
    #                 self.lr_scheduler.step()

    #     # update all losses
    #     for loss_name in self.config.writer.loss_names:
    #         metrics.update(loss_name, batch[loss_name].item())

    #     # update metrics
    #     for met in metric_funcs:
    #         if met.is_global:
    #             num, denum = met(**batch)
    #             metrics.update_global(met.name, num, denum)
    #         else:
    #             metrics.update(met.name, met(**batch))

    #     return batch, outputs['logits'][:, 1].detach().cpu(), batch['labels'].detach().cpu()


    def process_batch(self, batch, metrics: MetricTracker, gradient_accum_steps: int = 16):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step with optional gradient accumulation.

        Args:
            batch (dict): dict-based batch from the dataloader.
            metrics (MetricTracker): MetricTracker object.
            gradient_accum_steps (int): number of steps to accumulate gradients
                before performing an optimizer update.

        Returns:
            batch (dict), logits, labels
        """

        # initialize gradient step counter if not present
        if not hasattr(self, "_grad_step"):
            self._grad_step = 0

        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]

            # Only zero grad if starting a new accumulation cycle
            if self._grad_step % gradient_accum_steps == 0:
                self.optimizer.zero_grad()
    
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            outputs = self.model(**batch)
            batch.update(outputs)

            all_losses = self.criterion(**batch)
            batch.update(all_losses)

        if self.is_train:
            # scale loss for gradient accumulation
            loss = batch["loss"] / gradient_accum_steps
            self.scaler.scale(loss).backward()

            self._grad_step += 1  # update counter

            # perform optimizer update only every gradient_accum_steps steps
            if self._grad_step % gradient_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                self._clip_grad_norm()
                metrics.update("grad_norm", self.scaler.get_scale())
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

        # update all losses
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        # update metrics
        for met in metric_funcs:
            if met.is_global:
                num, denum = met(**batch)
                metrics.update_global(met.name, num, denum)
            else:
                metrics.update(met.name, met(**batch))
        return batch, outputs['logits'][:, 1].detach().cpu(), batch['labels'].detach().cpu()


    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            # Log Stuff
            pass
        else:
            # Log Stuff
            pass
