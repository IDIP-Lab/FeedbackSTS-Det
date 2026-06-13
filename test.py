import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import time
import datetime
import numpy as  np
import scipy.io as scio
import cv2

import torch
import torch.utils.data as Data
from collections import OrderedDict

from argparse import ArgumentParser

from model.FeedbackSTS import FeedbackSTS
from utils.dataset import IRSatVideoLoader, MIRSDTLoader
from utils.utils import test_save_dir_settings, create_dir, normalized, save_image

def parse_args():
    #
    # Setting parameters
    #
    parser = ArgumentParser(description='Implement of pytorch multi frame test')

    #
    # Dataset parameters
    #
    parser.add_argument("--dataset_names", default="IRSatVideo-LEO", type=str, nargs='+', 
                    help = "dataset_name: 'IRSatVideo-LEO', 'NUDT-MIRSDT-NEW-v2'")
    
    #
    # Testing parameters
    #
    parser.add_argument("--dataset_dir", default='../dataset/', type=str, help="test_dataset_dir")
    parser.add_argument("--seq_len", default=20, type = int, help="The length of the sequence")
    parser.add_argument("--inch", default=32, type = int, help="The input channel of the network.")
    parser.add_argument('--gpu', type=str, default='0', help='GPU number')
    parser.add_argument("--patchSize", default=256, type = int, help="[For MIRSTD only.]The length of the sequence")

    #
    # Pth parameters
    #
    parser.add_argument("--pth_paths", default=None, type=str, nargs='+', help="checkpoint dir, default=None")

    #
    # Save parameters
    #
    parser.add_argument('--base_save_dir', type=str, default='../result/pic_rst/', help='saving dir')
    parser.add_argument("--save_img", default=1, type=int, help="save image of or not")
    parser.add_argument("--save_mat", default=0, type=int, help="save image of or not")

    args = parser.parse_args()
    
    return args

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

class Tester(object):
    def __init__(self, args):

        self.args = test_save_dir_settings(args)
        
        ## GPU
        # if torch.cuda.is_available():
        #     os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        # self.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            torch.cuda.set_device(int(args.gpu))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ## model
        self.net = FeedbackSTS()
        self.net = self.net.to(self.device)

        ## Dataset
        if args.dataset_name == 'IRSatVideo-LEO':
            self.test_set = IRSatVideoLoader(dataset_dir = args.dataset_dir, 
                        dataset_name = args.dataset_name, seq_len = args.seq_len,
                        patch_size = args.patchSize, mode = "test")
        elif "MIRSDT" in args.dataset_name:
            self.test_set = MIRSDTLoader(dataset_dir = args.dataset_dir, 
                        dataset_name = args.dataset_name, seq_len = args.seq_len,
                        patch_size = args.patchSize, mode = "test")
        else:
            print("The dataset name does not exit.")
            raise NotImplementedError
        
        self.test_loader = Data.DataLoader(self.test_set, batch_size = 1, shuffle=False)
        self.max_iter = len(self.test_loader)
    
    def testing(self, pth_path):
        ## load pth path
        #self.net.load_state_dict(torch.load(pth_path, map_location = self.device))
        checkpoint = torch.load(pth_path, map_location = self.device)
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            new_key = k.replace("module.", "")  # 去掉 "module." 前缀
            new_state_dict[new_key] = v
        self.net.load_state_dict(new_state_dict)
        self.net.eval()

        ## set path
        pth_path_name = (pth_path.split(".")[-3]).split("/")[-1]
        save_pth_dir = self.args.dataset_save_dir + pth_path_name + '/'
        create_dir(save_pth_dir)
        
        if self.args.save_img > 0:
            save_img_sequence_dir = save_pth_dir + "img/"
            create_dir(save_img_sequence_dir)
        if self.args.save_mat > 0:
            save_mat_sequence_dir = save_pth_dir + "mat/"
            create_dir(save_mat_sequence_dir)

        ## inform
        base_log = "Iter: [{:d}/{:d}] || Cost Time: {} || Estimated Time: {}"
        iter_num = 0
        start_time = time.time()

        with torch.no_grad():
            for idx_iter, (img, _, seq_path_names, H_list, W_list) in enumerate(self.test_loader):
                img = img.to(self.device)
                pred = self.net(img)
                H = H_list[0].item()
                W = W_list[0].item()

                iter_num += 1

                cost_string = str(datetime.timedelta(seconds=int(time.time() - start_time)))
                eta_seconds = ((time.time() - start_time) / iter_num) * (self.max_iter - iter_num)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                
                ## Save the result
                _, _, FrameNum, resultH, resultW = pred.shape
                for ii in range(FrameNum):
                    img_save = np.array(pred[0,0, ii, :,:].cpu().detach())
                    realH, realW = img_save.shape
                    if (H != realH) or (W != realW):
                        img_save = cv2.resize(img_save, (W, H), interpolation = cv2.INTER_LINEAR)

                    seq_name, img_name = self.get_seq_img_name(seq_path_names[ii][0])
                    if self.args.save_img > 0:
                        save_img_dir = save_img_sequence_dir + seq_name + "/"
                        create_dir(save_img_dir)
                        img_save = (normalized(img_save) * 255.0).astype(np.uint8)
                        save_path = save_img_dir + img_name + ".png"
                        save_image(img_save, save_path)

                    if self.args.save_mat > 0:
                        save_mat_dir = save_mat_sequence_dir + seq_name + "/"
                        create_dir(save_mat_dir)
                        img_save = normalized(img_save)
                        save_path = save_mat_dir + img_name[0] + ".mat"
                        save_dict = {"rstImg": img_save}
                        scio.savemat(save_path, save_dict)
                
                print(base_log.format(iter_num, self.max_iter, cost_string, eta_string))
    
    def get_seq_img_name(self, seq_path_name):
        seq_path_name_list = seq_path_name.split("/")
        seq_name = seq_path_name_list[0]
        img_name = seq_path_name_list[1].split(".")[0]
        return seq_name, img_name

if __name__ == '__main__':
    args = parse_args()

    dataset_names = names_standard_format(args.dataset_names)
    pth_paths = names_standard_format(args.pth_paths)

    for dataset_name in dataset_names:
        args.dataset_name = dataset_name

        tester = Tester(args)
        for pth_path in pth_paths:
            tester.testing(pth_path)
                