# Slightly modified code from here:
# https://github.com/qubvel/segmentation_models.pytorch
# https://github.com/gasparian/multiclass-semantic-segmentation

import os
import numpy as np
import logging

from tensorboardX import SummaryWriter

import torch

class Meter:

    '''A meter to keep track of iou and dice scores throughout an epoch'''

    def __init__(self, root_result_dir="", base_threshold=.5, get_class_metric=False):
        self.base_threshold = base_threshold # threshold
        self.get_class_metric = get_class_metric
        # tensorboard logging
        if root_result_dir:
            self.tb_log = SummaryWriter(log_dir=os.path.join(root_result_dir, 'tensorboard'))
        self.reset_dicts()

    def reset_dicts(self):
        self.base_dice_scores = {"train":[], "val":[]}
        self.dice_neg_scores = {"train":[], "val":[]}
        self.dice_pos_scores = {"train":[], "val":[]}
        self.iou_scores = {"train":[], "val":[]}

    def predict(self, X):
        '''X is sigmoid output of the model'''
        X_p = np.copy(X)
        preds = (X_p > self.base_threshold).astype('uint8')
        return preds

    def metric(self, probability, truth):
        '''Calculates dice of positive and negative images seperately'''
        '''probability and truth must be torch tensors'''
        batch_size = len(truth)
        with torch.no_grad():
            probability = (probability > self.base_threshold).float()
            truth = (truth > 0.5).float()

            p = probability.view(batch_size, -1) # size 변경:
            t = truth.view(batch_size, -1)
            assert(p.shape == t.shape)

            intersection = (p*t).sum(-1)
            union = (p+t).sum(-1)

            t_sum = t.sum(-1)
            p_sum = p.sum(-1)
            neg_index = torch.nonzero(t_sum == 0)
            pos_index = torch.nonzero(t_sum >= 1)

            neg = (p_sum == 0).float()
            dice_pos = (2 * intersection) / (union + 1e-7)
            iou_pos = intersection / (union + 1e-7)

            neg = neg[neg_index]
            dice_pos = dice_pos[pos_index]
            iou_pos = iou_pos[pos_index]

            dice = torch.cat([dice_pos, neg])
            iou = torch.cat([iou_pos, neg])

            neg = np.nan_to_num(neg.mean().item(), 0)
            dice_pos = np.nan_to_num(dice_pos.mean().item(), 0)

            dice = dice.mean().item()
            iou = iou.mean().item()

            num_neg = len(neg_index)
            num_pos = len(pos_index)

            dice = {"dice_all": dice}

            if self.get_class_metric:
                num_classes = probability.shape[1]
                for c in range(num_classes):
                    iflat = probability[:, c,...].view(batch_size, -1)
                    tflat = truth[:, c,...].view(batch_size, -1)
                    intersection = (iflat * tflat).sum()
                    dice[str(c)] = ((2. * intersection) / (iflat.sum() + tflat.sum() + 1e-7)).item()

        return iou, dice, neg, dice_pos, num_neg, num_pos

    def update(self, phase, targets, outputs):
        """updates metrics lists every iteration"""
        iou, dice, dice_neg, dice_pos, _, _ = self.metric(outputs, targets)
        self.base_dice_scores[phase].append(dice)
        self.dice_pos_scores[phase].append(dice_pos)
        self.dice_neg_scores[phase].append(dice_neg)
        self.iou_scores[phase].append(iou)

    def get_metrics(self, phase):
        """averages computed metrics over the epoch"""
        dice = {}
        l = len(self.base_dice_scores[phase])
        for i, d in enumerate(self.base_dice_scores[phase]):
            for k in d:
                if k not in dice:
                    dice[k] = 0
                dice[k] += d[k] / l

        dice_neg = np.mean(self.dice_neg_scores[phase])
        dice_pos = np.mean(self.dice_pos_scores[phase])
        dices = [dice, dice_neg, dice_pos]
        iou = np.nanmean(self.iou_scores[phase])
        return dices, iou

    def epoch_log(self, phase, epoch_loss, itr):
        '''logging the metrics at the end of an epoch'''
        dices, iou = self.get_metrics(phase)
        dice, dice_neg, dice_pos = dices
        message = "Phase: %s | Loss: %0.4f | IoU: %0.4f | dice: %0.4f | dice_neg: %0.4f | dice_pos: %0.4f" \
                  % (phase, epoch_loss, iou, dice["dice_all"], dice_neg, dice_pos)
        logging.info(message)

        self.tb_log.add_scalar(f'{phase}_dice', dice["dice_all"], itr)
        self.tb_log.add_scalar(f'{phase}_dice_neg', dice_neg, itr)
        self.tb_log.add_scalar(f'{phase}_dice_pos', dice_pos, itr)
        self.tb_log.add_scalar(f'{phase}_iou', iou, itr)
        return dice, iou


class BCEDiceLoss:
    def __init__(self, bce_weight=1., weight=None, eps=1e-7,
                 smooth=.0, class_weights=[], threshold=0., activate=False):

        self.bce_weight = bce_weight
        self.eps = eps
        self.smooth = smooth
        self.threshold = threshold  # 0 or apply sigmoid and threshold > .5 instead
        self.activate = activate
        self.class_weights = class_weights
        self.nll = torch.nn.BCEWithLogitsLoss(weight=weight)

    def __call__(self, logits, true):
        loss = self.bce_weight * self.nll(logits, true)
        if self.bce_weight < 1.:
            dice_loss = 0.
            batch_size, num_classes = logits.shape[:2]
            if self.activate:
                logits = torch.sigmoid(logits)
            logits = (logits > self.threshold).float()
            for c in range(num_classes):
                iflat = logits[:, c, ...].view(batch_size, -1)
                tflat = true[:, c, ...].view(batch_size, -1)
                intersection = (iflat * tflat).sum()

                w = self.class_weights[c]
                dice_loss += w * ((2. * intersection + self.smooth) /
                                  (iflat.sum() + tflat.sum() + self.smooth + self.eps))
            loss -= (1 - self.bce_weight) * torch.log(dice_loss)

        return loss
