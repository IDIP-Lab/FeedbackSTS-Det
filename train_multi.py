import time
import datetime

import sys
import os

# ------------------------------------------------------------
# Must set CUDA_VISIBLE_DEVICES before any torch import to
# prevent PyTorch from occupying other GPUs.
# Parse --gpu from sys.argv manually to extract GPU indices.
# ------------------------------------------------------------
gpu_custom = "0"
for i, arg in enumerate(sys.argv):
    if arg == "--gpu" and i + 1 < len(sys.argv):
        gpu_custom = sys.argv[i + 1]
        break
if "," in gpu_custom or " " in gpu_custom:
    gpu_custom = gpu_custom.replace(" ", ",")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_custom
elif gpu_custom:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_custom

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.utils.data as Data

from tensorboardX import SummaryWriter

from model.FeedbackSTS import FeedbackSTS
from utils.dataset import IRSatVideoLoader, MIRSDTLoader
from utils.loss import SoftIoULoss
from utils.logger import setup_logger
from utils.utils import get_optimizer, train_save_dir_settings, save_checkpoint


def parse_args():
    #
    # Setting parameters
    #
    parser = ArgumentParser(description='Implement of pytorch multi frame train')

    #
    # Dataset parameters
    #
    parser.add_argument("--dataset_names", default="IRSatVideo-LEO", type=str, nargs='+',
                        help="dataset_name: 'IRSatVideo-LEO'")

    #
    # Training parameters
    #
    parser.add_argument("--dataset_dir", default='./dataset/', type=str, help="train_dataset_dir")
    parser.add_argument("--seq_len", default=9, type=int, help="The length of the sequence")
    parser.add_argument("--sample_rate", type=int, default=5,
                        help="[for IRSatVideo-LEO] the training rate for IRSatVideo-LEO")
    parser.add_argument("--sample_space", default=1, type=int, help="[for MIRSTD]the space between the sequences")
    parser.add_argument("--batchSize", type=int, default=3, help="Training batch sizse")
    parser.add_argument("--precision", type=str, default='32F', help="Training Precision. 16F, 32F")
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument("--patchSize", type=int, default=256, help="Training patch size")
    parser.add_argument('--gpu', type=str, default='0',
                        help="GPU number(s): single GPU like '3', multi GPU like '1,3,5' or '1 3 5'")
    parser.add_argument('--seed', type=int, default=42, help='seed: 0, 42, 1307等')
    parser.add_argument("--optimizer_name", default='Adam', type=str, help="optimizer name: Adam, Adagrad, SGD")
    parser.add_argument("--optimizer_settings", default={'lr': 5e-4}, type=dict, help="optimizer settings")
    parser.add_argument("--scheduler_name", default='MultiStepLR', type=str, help="scheduler name: MultiStepLR")
    parser.add_argument("--scheduler_settings", default={'step': [5, 10, 15, 20, 25, 30], 'gamma': 0.5}, type=dict,
                        help="scheduler settings")

    #
    # Save parameters
    #
    parser.add_argument('--log-per-epoch', type=int, default=1, help='interval of logging between epochs')
    parser.add_argument('--log-per-iter', type=int, default=1, help='interval of logging between iters')
    parser.add_argument('--save_iter_step', type=int, default=5, help='save model per step iters')
    parser.add_argument('--base-dir', type=str, default='../result/', help='saving dir')

    args = parser.parse_args()

    # seed
    if args.seed != 0:
        set_seeds(args.seed)

    return args


def set_seeds(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def names_standard_format(original_names):
    if (isinstance(original_names, list)):
        if (len(original_names) == 1):
            names = original_names[0].split(',') if ',' in original_names[0] else original_names[0].split()
        else:
            names = original_names
    elif (original_names == None):
        return original_names
    else:
        names = original_names.split(",")
    return names


class Trainer(object):
    def __init__(self, args):

        self.iter_num = 0

        ## Save folders
        self.args = train_save_dir_settings(args=args)

        ## dataset
        if args.dataset_name == 'IRSatVideo-LEO':
            self.train_set = IRSatVideoLoader(dataset_dir=args.dataset_dir,
                                              dataset_name=args.dataset_name, seq_len=args.seq_len,
                                              sample_space=args.sample_space, patch_size=args.patchSize)
        elif "MIRSDT" in args.dataset_name:
            self.train_set = MIRSDTLoader(dataset_dir=args.dataset_dir,
                                          dataset_name=args.dataset_name, seq_len=args.seq_len,
                                          sample_space=args.sample_space, patch_size=args.patchSize)
        else:
            print("The dataset name does not exit.")
            raise NotImplementedError

        self.train_loader = Data.DataLoader(self.train_set, batch_size=args.batchSize, shuffle=True)
        self.iter_per_epoch = len(self.train_loader)
        self.max_iter = args.epochs * self.iter_per_epoch

        ## model
        self.net = FeedbackSTS()

        # ------------------------------------------------------------
        # GPU setup: support single & multiple GPUs
        # ------------------------------------------------------------
        gpu_list = names_standard_format(args.gpu)
        gpu_ids = [int(g) for g in gpu_list]

        if torch.cuda.is_available():
            if len(gpu_ids) > 1:
                # Multi-GPU: CUDA_VISIBLE_DEVICES already set at top of file,
                # GPUs are now seen as 0, 1, 2...
                self.device = torch.device("cuda:0")
                self.net = nn.DataParallel(self.net, device_ids=list(range(len(gpu_ids))))
            else:
                # Single GPU
                torch.cuda.set_device(gpu_ids[0])
                self.device = torch.device("cuda:{}".format(gpu_ids[0]))
        else:
            self.device = torch.device("cpu")

        self.net = self.net.to(self.device)

        # ------------------------------------------------------------
        # Half-precision initialization (AMP)
        # ------------------------------------------------------------
        self.amp_enabled = (self.args.precision == "16F")
        if self.amp_enabled:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None

        ## criterion
        self.cal_loss = SoftIoULoss()

        ## optimizer, scheduler
        self.optimizer, self.scheduler = get_optimizer(self.net, args.optimizer_name, args.scheduler_name,
                                                       args.optimizer_settings, args.scheduler_settings)

        ## SummaryWriter
        self.writer = SummaryWriter(log_dir=args.save_log_dir)
        self.writer.add_text(args.dir_name, 'Args:%s, ' % args)

        ## log info
        self.model_name = "FeedBackSTS-Det"
        self.logger = setup_logger(self.model_name, args.save_log_dir, 0, filename='log.txt')
        self.logger.info(args)
        self.logger.info("Using device: {}".format(self.device))
        self.logger.info(self.model_name + " + " + args.dataset_name)

    def training(self):
        total_loss_list = []
        total_loss_epoch = []

        start_time = time.time()
        base_log = "Epoch-Iter: [{:d}/{:d}]-[{:d}/{:d}] || Lr: {:.6f} || Loss: {:.4f} || " \
                   "Cost Time: {} || Estimated Time: {}"

        for idx_epoch in range(self.args.epochs):
            for idx_iter, (img, gt_mask) in enumerate(self.train_loader):
                self.net.train()

                img = img.to(self.device)
                gt_mask = gt_mask.to(self.device)

                if img.shape[0] == 1:
                    continue

                # ------------------------------------------------------------
                # Forward pass with optional AMP autocast
                # ------------------------------------------------------------
                if self.amp_enabled:
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        pred = self.net(img)
                        loss = self.cal_loss(pred, gt_mask)
                else:
                    pred = self.net(img)
                    loss = self.cal_loss(pred, gt_mask)

                total_loss_epoch.append(loss.detach().cpu())

                # ------------------------------------------------------------
                # Backward pass with optional gradient scaling
                # ------------------------------------------------------------
                self.optimizer.zero_grad()
                if self.amp_enabled:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                self.iter_num += 1

                cost_string = str(datetime.timedelta(seconds=int(time.time() - start_time)))
                eta_seconds = ((time.time() - start_time) / self.iter_num) * (self.max_iter - self.iter_num)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                self.writer.add_scalar('Train Loss/', np.mean(loss.item()), self.iter_num)
                self.writer.add_scalar('Learning rate/', trainer.optimizer.param_groups[0]['lr'], self.iter_num)

                if self.iter_num % self.args.log_per_iter == 0:
                    self.logger.info(
                        base_log.format(idx_epoch + 1, self.args.epochs, self.iter_num % self.iter_per_epoch,
                                        self.iter_per_epoch, self.optimizer.state_dict()['param_groups'][0]['lr'],
                                        loss.item(), cost_string, eta_string))

                    # Flush logger to ensure output in tmux
                    for handler in self.logger.handlers:
                        handler.flush()

            self.scheduler.step()

            if (idx_epoch + 1) % self.args.log_per_epoch == 0:
                total_loss_list.append(float(np.array(total_loss_epoch).mean()))
                self.logger.info(time.ctime()[4:-5] + ' Epoch---%d, lr---%f, total_loss---%f' % (idx_epoch + 1,
                                                                                                 self.optimizer.state_dict()[
                                                                                                     'param_groups'][0][
                                                                                                     'lr'],
                                                                                                 total_loss_list[-1]))
                total_loss_epoch = []
                # Flush logger after epoch log
                for handler in self.logger.handlers:
                    handler.flush()

            if (idx_epoch + 1) % self.args.save_iter_step == 0:
                save_pth = self.args.dataset_save_dir \
                           + self.model_name + '_' \
                           + "SeqLen{:02d}".format(self.args.seq_len) + '_' \
                           + "{:02d}".format(idx_epoch + 1) + '.pth.tar'

                torch.save(self.net.state_dict(), save_pth)

            if (idx_epoch + 1) == self.args.epochs and (idx_epoch + 1) % 5 != 0:
                save_pth = self.args.dataset_save_dir \
                           + self.model_name + '_' \
                           + "SeqLen{:02d}".format(self.args.seq_len) + '_' \
                           + "{:02d}".format(idx_epoch + 1) + '.pth.tar'

                torch.save(self.net.state_dict(), save_pth)


if __name__ == '__main__':
    args = parse_args()
    dataset_names = names_standard_format(args.dataset_names)
    for dataset_name in dataset_names:
        args.dataset_name = dataset_name
        trainer = Trainer(args)
        trainer.training()

        ## 删除实例
        torch.cuda.empty_cache()