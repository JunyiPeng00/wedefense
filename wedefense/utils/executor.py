# Copyright (c) 2021 Hongji Wang (jijijiang77@gmail.com)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#               2025 Lin Zhang (partialspoof@gmail.com)
#                    Shuai Wang (wsstriving@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import logging

import tableprint as tp
import torch
import torchnet as tnt

from wedefense.dataset.dataset_utils import apply_cmvn, spec_aug


def train_epoch(dataloader,
                epoch_iter,
                model,
                criterion,
                optimizer,
                scheduler,
                margin_scheduler,
                epoch,
                logger,
                scaler,
                device,
                configs,
                wandb_log=None):
    """Train the model for one epoch.

    Args:
        dataloader: Training dataloader
        epoch_iter: Number of iterations per epoch
        model: Model to train
        criterion: Loss criterion
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        margin_scheduler: Margin scheduler
        epoch: Current epoch number
        logger: Logger
        scaler: Gradient scaler for mixed precision training
        device: Device to run training on
        configs: Configuration dictionary

    Returns:
        tuple: (training loss, training accuracy, last batch index)
    """
    model.train()
    # By default use average pooling
    loss_meter = tnt.meter.AverageValueMeter()
    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)

    frontend_type = configs['dataset_args'].get('frontend', 'fbank')
    nan_batches = 0
    for i, batch in enumerate(dataloader):
        cur_iter = (epoch - 1) * epoch_iter + i
        scheduler.step(cur_iter)
        margin_scheduler.step(cur_iter)

        targets = batch['label']
        targets = targets.long().to(device)  # (B)
        if frontend_type == 'fbank' or frontend_type.startswith('lfcc'):
            features = batch['feat']  # (B,T,F)
            features = features.float().to(device)
        else:  # 's3prl'
            wavs = batch['wav']  # (B,1,W)
            wavs = wavs.squeeze(1).float().to(device)  # (B,W)
            wavs_len = torch.LongTensor([wavs.shape[1]]).repeat(
                wavs.shape[0]).to(device)  # (B)
            with torch.cuda.amp.autocast(enabled=configs['enable_amp']):
                features, _ = model.module.frontend(wavs, wavs_len)

        with torch.cuda.amp.autocast(enabled=configs['enable_amp']):
            # apply cmvn
            if configs['dataset_args'].get('cmvn', True):
                features = apply_cmvn(
                    features, **configs['dataset_args'].get('cmvn_args', {}))
            # spec augmentation
            if configs['dataset_args'].get('spec_aug', False):
                features = spec_aug(features,
                                    **configs['dataset_args']['spec_aug_args'])

            outputs = model(features)  # (embed_a,embed_b) in most cases
            embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
            outputs = model.module.projection(embeds, targets)
            if isinstance(outputs, tuple):
                outputs, loss = outputs
            else:
                loss = criterion(outputs, targets)

            if configs.get('ce_per_sample_fp32', False):
                # replaces the criterion value: plain cross-entropy per
                # sample on float32 logits, mean over the batch
                loss = torch.nn.functional.cross_entropy(
                    outputs.float(), targets, reduction='none').mean()

        # a non-finite loss on any rank: every rank drops the gradients
        finite = torch.isfinite(loss.detach()).float()
        if torch.distributed.is_available(
        ) and torch.distributed.is_initialized():
            torch.distributed.all_reduce(finite,
                                         op=torch.distributed.ReduceOp.MIN)
        if finite.item() < 1:
            nan_batches += 1
            if nan_batches <= 20:
                logging.warning(
                    "non-finite loss at epoch %d batch %d; keys=%s", epoch, i,
                    str(batch.get('key', batch.get('keys', '?')))[:300])
            optimizer.zero_grad()
            continue

        # loss, acc
        loss_meter.add(loss.item())
        acc_meter.add(outputs.float().cpu().detach().numpy(),
                      targets.cpu().numpy())

        # update the model every accum_grad batches
        accum = int(configs.get('accum_grad', 1))
        if i % accum == 0:
            optimizer.zero_grad()
        step_now = (i + 1) % accum == 0 or (i + 1) == epoch_iter
        # no gradient synchronisation between ranks before the step
        sync_ctx = contextlib.nullcontext() if (
            step_now or not hasattr(model, 'no_sync')) else model.no_sync()
        with sync_ctx:
            # scaler does nothing here if enable_amp=False
            scaler.scale(loss / accum).backward()
        if step_now:
            scaler.step(optimizer)
            scaler.update()

        if (wandb_log):
            wandb_log.log({
                "learning_rate": scheduler.get_lr(),
                "train/loss": loss_meter.value()[0],
                "train/acc": acc_meter.value()[0]
            })
        # log
        if (i + 1) % configs['log_batch_interval'] == 0:
            logger.info(
                tp.row((epoch, i + 1, scheduler.get_lr(),
                        margin_scheduler.get_margin()) +
                       (loss_meter.value()[0], acc_meter.value()[0]),
                       width=10,
                       style='grid'))

        if (i + 1) == epoch_iter:
            break

    # Final log for this epoch
    logger.info(
        tp.row(
            (epoch, i + 1, scheduler.get_lr(), margin_scheduler.get_margin()) +
            (loss_meter.value()[0], acc_meter.value()[0]),
            width=10,
            style='grid'))


def val_epoch(val_dataloader,
              val_iter,
              model,
              criterion,
              device,
              configs,
              wandb_log=None,
              spoof_id=None,
              return_metrics=False):
    """Validate the model on the validation set.

    Args:
        val_dataloader: Validation dataloader
        model: Model to validate
        criterion: Loss criterion
        device: Device to run validation on
        configs: Configuration dictionary
        spoof_id: Index of the spoof class
        return_metrics: Also return val_metrics() of the spoof posteriors

    Returns:
        tuple: (validation loss, validation accuracy[, val_metrics() dict])
    """
    model.eval()
    val_loss_meter = tnt.meter.AverageValueMeter()
    val_acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
    # per-utterance spoof posteriors, for Macro-F1 at threshold 0.5
    v_keys, v_prob, v_true = [], [], []

    frontend_type = configs['dataset_args'].get('frontend', 'fbank')
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            targets = batch['label']
            targets = targets.long().to(device)  # (B)
            if frontend_type == 'fbank' or frontend_type.startswith('lfcc'):
                features = batch['feat']  # (B,T,F)
                features = features.float().to(device)
            else:  # 's3prl'
                wavs = batch['wav']  # (B,1,W)
                wavs = wavs.squeeze(1).float().to(device)  # (B,W)
                wavs_len = torch.LongTensor([wavs.shape[1]]).repeat(
                    wavs.shape[0]).to(device)  # (B)
                with torch.cuda.amp.autocast(enabled=configs['enable_amp']):
                    features, _ = model.module.frontend(wavs, wavs_len)

            with torch.cuda.amp.autocast(enabled=configs['enable_amp']):
                # apply cmvn
                if configs['dataset_args'].get('cmvn', True):
                    features = apply_cmvn(
                        features,
                        **configs['dataset_args'].get('cmvn_args', {}))

                outputs = model(features)  # (embed_a,embed_b) in most cases
                embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
                outputs = model.module.projection(embeds, targets)
                if isinstance(outputs, tuple):
                    outputs, loss = outputs
                else:
                    loss = criterion(outputs, targets)

            # loss, acc
            val_loss_meter.add(loss.item())
            val_acc_meter.add(outputs.float().cpu().detach().numpy(),
                              targets.cpu().numpy())
            if return_metrics and spoof_id is not None:
                probs = torch.softmax(outputs.float(), dim=-1)[:, spoof_id]
                v_prob.extend(probs.cpu().tolist())
                v_true.extend((targets == spoof_id).long().cpu().tolist())
                v_keys.extend(list(batch.get('key', [''] * len(probs))))
            if (wandb_log):
                wandb_log.log({
                    "val/loss": val_loss_meter.value()[0],
                    "val/acc": val_acc_meter.value()[0]
                })

            if (i + 1) == val_iter:
                break

    if not return_metrics:
        return val_loss_meter.value()[0], val_acc_meter.value()[0]
    return (val_loss_meter.value()[0], val_acc_meter.value()[0],
            val_metrics(v_keys, v_prob, v_true,
                        configs.get('val_group_weights')))


def val_metrics(keys, prob_spoof, is_spoof, group_weights=None, threshold=0.5):
    """Macro-F1 at a fixed threshold, overall and weighted over groups.

    Args:
        keys: utterance keys
        prob_spoof: P(spoof) per utterance
        is_spoof: reference per utterance (1 = spoof)
        group_weights: {group: weight}; the group of an utterance is the
            first path component of its key. Groups without utterances are
            skipped and the weights renormalised.

    Returns:
        dict: 'macro_f1', 'groups' {group: (count, Macro-F1)} and
            'weighted_macro_f1' (the overall value if there is no group)
    """
    from sklearn.metrics import f1_score
    import numpy as np
    if not prob_spoof:
        return {}
    p = np.asarray(prob_spoof)
    y = np.asarray(is_spoof)
    pred = (p >= threshold).astype(int)
    out = {'macro_f1': float(f1_score(y, pred, average='macro', labels=[0, 1]))}
    groups = {}
    if group_weights:
        for k, yy, pp in zip(keys, y, pred):
            g = str(k).split('/')[0]
            if g in group_weights:
                groups.setdefault(g, [[], []])
                groups[g][0].append(yy)
                groups[g][1].append(pp)
    if groups:
        wsum, acc = 0.0, 0.0
        for g, (yy, pp) in groups.items():
            f = float(f1_score(yy, pp, average='macro', labels=[0, 1]))
            out.setdefault('groups', {})[g] = (len(yy), round(f, 4))
            acc += group_weights[g] * f
            wsum += group_weights[g]
        out['weighted_macro_f1'] = acc / wsum
    else:
        out['weighted_macro_f1'] = out['macro_f1']
    return out
