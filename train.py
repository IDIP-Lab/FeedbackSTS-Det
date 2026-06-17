import time
import datetime

import numpy as np
from argparse import ArgumentParser

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

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
                    help = "dataset_name: 'IRSatVideo-LEO', 'NUDT-MIRSDT-NEW-v2'")

    #
    # Training parameters
    #
    parser.add_argument("--dataset_dir", default = '../dataset/', type=str, help="train_dataset_dir")
    parser.add_argument("--seq_len", default=20, type = int, help="The length of the sequence")
    parser.add_argument("--sample_rate", type=int, default=5, help="[for IRSatVideo-LEO] the training rate for IRSatVideo-LEO")
    parser.add_argument("--sample_space", default=1, type = int, help="[for MIRSTD]the space between the sequences")
    parser.add_argument("--batchSize", type=int, default = 3 , help="Training batch sizse")
    parser.add_argument("--precision", type=str, default = '32F', help="Training Precision. 16F, 32F")
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument("--patchSize", type=int, default=256, help="Training patch size")
    parser.add_argument('--gpu', type=str, default='0', #nargs='+',
                        help= "GPU number: '0 2 3'")
    parser.add_argument('--seed', type=int, default=3047, help='seed: 0, 42, 1307, 3047等')
    parser.add_argument("--optimizer_name", default='Adam', type=str, help="optimizer name: Adam, Adagrad, SGD")
    parser.add_argument("--optimizer_settings", default={'lr': 1e-5}, type=dict, help="optimizer settings")
    parser.add_argument("--scheduler_name", default='MultiStepLR', type=str, help="scheduler name: MultiStepLR")
    parser.add_argument("--scheduler_settings", default={'step': [5, 10, 15, 20, 25, 30], 'gamma': 0.5}, type=dict, help="scheduler settings")

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
    # torch.backends.cudnn.deterministic = True

def names_standard_format(original_names):
    if(isinstance(original_names, list)):
        if(len(original_names)==1):
            names = original_names[0].split(" ")
        else:
            names = original_names
    elif (original_names == None):
        return original_names
    else:
        names = original_names.split(" ")
    return names

class Trainer(object):
    def __init__(self, args):

        self.iter_num = 0

        ## Save folders
        self.args = train_save_dir_settings(args = args)

        ## dataset
        if args.dataset_name == 'IRSatVideo-LEO':
            self.train_set = IRSatVideoLoader(dataset_dir = args.dataset_dir, 
                        dataset_name = args.dataset_name, seq_len = args.seq_len,
                        sample_space = args.sample_space, patch_size = args.patchSize)
        elif "MIRSDT" in args.dataset_name:
            self.train_set = MIRSDTLoader(dataset_dir = args.dataset_dir, 
                        dataset_name = args.dataset_name, seq_len = args.seq_len,
                        sample_space = args.sample_space, patch_size = args.patchSize)
        else:
            print("The dataset name does not exit.")
            raise NotImplementedError

        self.train_loader = Data.DataLoader(self.train_set, batch_size=args.batchSize, shuffle=True)
        self.iter_per_epoch = len(self.train_loader)
        self.max_iter = args.epochs * self.iter_per_epoch

        ## GPU
        if torch.cuda.is_available():
            torch.cuda.set_device(int(args.gpu))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ## model
        self.net = FeedbackSTS()

        self.net = self.net.to(self.device)

        # ------------------------------------------------------------
        # Half-precision initialization
        # ------------------------------------------------------------
        self.amp_enabled = (self.args.precision == "16F")
        if self.amp_enabled:
            # AMP scaler for gradient scaling (prevents underflow in half)
            #self.scaler = torch.cuda.amp.GradScaler()
            self.scaler = torch.amp.GradScaler('cuda')
            # Ensure the DCN2 CUDA extension is properly initialized for half
            # (no need to call net.half() — AMP autocast handles it dynamically)
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

                if img.shape[0] == 1: # batch size 为 1 又有什么影响?
                    continue

                # ------------------------------------------------------------
                # Forward pass with optional AMP autocast
                # ------------------------------------------------------------
                if self.amp_enabled:
                    #with torch.cuda.amp.autocast(dtype=torch.float16):
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
                    self.logger.info(base_log.format(idx_epoch + 1, self.args.epochs, self.iter_num % self.iter_per_epoch,
                                                     self.iter_per_epoch, self.optimizer.state_dict()['param_groups'][0]['lr'],
                                                     loss.item(), cost_string, eta_string))

            self.scheduler.step()

            if (idx_epoch + 1) % self.args.log_per_epoch == 0:
                total_loss_list.append(float(np.array(total_loss_epoch).mean()))
                self.logger.info(time.ctime()[4:-5] + ' Epoch---%d, lr---%f, total_loss---%f' % (idx_epoch + 1, 
                                    self.optimizer.state_dict()['param_groups'][0]['lr'], total_loss_list[-1]))
                total_loss_epoch = []

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
