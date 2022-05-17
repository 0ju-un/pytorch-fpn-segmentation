import time

import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm

from loss import *

class Trainer(object):
    '''Basic functionality for models fitting'''

    def __init__(self, model=None, dataloaders=None, optimizer=None,
                 lr=0.001, batch_size=25, num_epochs=0,
                 model_path="", load_checkpoint=""):
        # Initialize logger
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s : %(levelname)s : %(message)s'
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.load_checkpoint=load_checkpoint

        self.weights_decay= 0.0
        self.bce_loss_weight = .7
        self.class_weights= [0.333, 0.333, 0.333]
        self.base_threshold = .0
        self.activate = False
        self.scheduler_patience = 3
        self.accumulation_batches = 1
        self.key_metric = "iou"

        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.net = model
        self.net.to(self.device)

        ############################################################################################

        self.best_metric = float("inf")
        self.phases = ["train", "val"]
        # torch.set_default_tensor_type("torch.cuda.FloatTensor")
        # self.freeze_flag = True
        self.start_epoch = 0
        cudnn.benchmark = True

        self.dataloaders = dataloaders
        self.criterion = BCEDiceLoss(bce_weight=self.bce_loss_weight, class_weights=self.class_weights,
                                     threshold=self.base_threshold, activate=self.activate)
        self.optimizer = optimizer(self.net.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=self.scheduler_patience, verbose=True)

        self.losses = {phase: [] for phase in self.phases}
        self.iou_scores = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}

        self.meter = Meter(self.model_path, self.base_threshold)

        if self.load_checkpoint:
            self.load_model(ckpt_name=self.load_checkpoint)

        self.accumulation_steps = self.batch_size * self.accumulation_batches

        # number of workers affect the GPU performance if the preprocessing too intensive (resizes \ augs)
        self.num_workers = max(2, self.batch_size // 2)
        # self.num_workers = self.batch_size

        # logging.info(f"Trainer initialized on {len(self.devices_ids)} devices!")

    def load_model(self, ckpt_name="best_model.pth"):
        """Loads full model state and basic training params"""
        path = "/".join(ckpt_name.split("/")[:-1])
        chkpt = torch.load(ckpt_name)
        self.start_epoch = chkpt['epoch']
        self.best_metric = chkpt['best_metric']

        self.net.load_state_dict(chkpt['state_dict'])
        self.optimizer.load_state_dict(chkpt['optimizer'])

        # if self.load_optimizer_state:
        #     self.optimizer.load_state_dict(chkpt['optimizer'])
        logging.info("******** State loaded ********")

    def forward(self, images, targets):
        """allocate data and runs forward pass through the network"""
        # send all variables to selected device
        images = images.to(self.device)
        masks = targets.to(self.device)
        # compute loss
        outputs = self.net(images)
        loss = self.criterion(outputs, masks)
        return loss, outputs

    def weightsdecay(self):
        """adjust learning rate and weights decay -- 가중치 감쇠"""
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                param.data = param.data.add(-1.*self.weights_decay * param_group['lr'], param.data)


    def iterate(self, epoch, phase):
        """main method for traning: creates metric aggregator, updates the model params"""

        self.meter.reset_dicts()
        start = time.strftime("%H:%M:%S")
        logging.info(f"Starting epoch: {epoch} | phase: {phase} | time: {start}")
        self.net.train(phase == "train")

        dataloader = self.dataloaders[phase]

        running_loss = 0.0
        total_batches = len(dataloader)
        self.optimizer.zero_grad()
        tk0 = tqdm(dataloader, total=total_batches)
        # if self.freeze_n_iters:
        #     self.freeze_encoder()

        for itr, batch in enumerate(tk0):

            # if itr == (self.freeze_n_iters - 1):
            #     self.freeze_flag = False
            #     self.freeze_encoder()
            images, targets = batch
            loss, outputs = self.forward(images, targets)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                if (itr + 1) % self.accumulation_steps == 0:
                    if self.weights_decay > 0:
                        self.weightsdecay()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            running_loss += loss.item()
            if self.activate:
                outputs = torch.sigmoid(outputs)
            outputs = outputs.detach().cpu()
            self.meter.update(phase, targets, outputs)
            running_loss_tick = (running_loss * self.accumulation_steps) / (itr + 1)

            self.meter.tb_log.add_scalar(f'{phase}_loss', running_loss_tick, itr + total_batches * epoch)
            tk0.set_postfix(loss=(running_loss_tick))

        last_itr = itr + total_batches * epoch # 뭔가 이상한데

        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        dice, iou = self.meter.epoch_log(phase, epoch_loss, last_itr)

        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice["dice_all"])
        self.iou_scores[phase].append(iou)

        torch.cuda.empty_cache()
        return epoch_loss, dice, iou, last_itr

    def get_current_lr(self):
        for i, param_group in enumerate(self.optimizer.param_groups):
            return float(param_group['lr'])

    def start(self):
        """Runs training loop and saves intermediate state"""

        for epoch in range(self.start_epoch, self.num_epochs + self.start_epoch):
            _, __, ___, last_itr = self.iterate(epoch, "train") # epoch_loss, dice, iou, last_itr
            self.meter.tb_log.add_scalar(f'learning_rate', self.get_current_lr(), last_itr)

            state = {
                "epoch": epoch,
                "best_metric": self.best_metric,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }

            with torch.no_grad():
                val_loss, val_dice, val_iou, __ = self.iterate(epoch, "val")
                key_metrics = {"loss": val_loss, "dice": (-1. * val_dice["dice_all"]), "iou": (-1. * val_iou)}  # -1* for the scheduler
                self.scheduler.step(val_loss)

            is_last_epoch = epoch == (self.num_epochs - 1)
            if key_metrics[self.key_metric] < self.best_metric or is_last_epoch:
                logging.info("******** Saving state ********")

                state["val_IoU"] = val_iou
                state["val_dice"] = val_dice
                state["val_loss"] = val_loss
                state["best_metric"] = self.best_metric = key_metrics[self.key_metric]

                ckpt_name = "best_model"
                if is_last_epoch:
                    ckpt_name = "last_model"
                torch.save(state, f"{self.model_path}/{ckpt_name}.pth")