import os
import numpy as np
import cv2
import torch

from argparse import ArgumentParser

import scipy.io as scio
from sklearn.metrics import auc

from evaluation.mIoU import mIoU
from evaluation.roc_curve import ROCMetric
from evaluation.pd_fa import PD_FA
from evaluation.my_pd_fa import my_PD_FA
from evaluation.TPFNFP import SegmentationMetricTPFNFP

from utils.utils import load_image, create_dir, normalized

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

def get_dataset_name(dir_name):
    dataset_name_list = ["IRSatVideo-LEO", "NUDT-MIRSDT-NEW-v2"]

    dir_name_list = dir_name.split("/")
    if dir_name_list[-1] == "":
        net_seqLen_name = dir_name_list[-2] 
    else:
        net_seqLen_name = dir_name_list[-1] 

    flag = False
    for dataset_name in dataset_name_list:
        if dataset_name in dir_name:
            dataset_name = dataset_name
            flag = True
            break

    if not flag:
        raise ValueError("Unknown dataset_name")

    return dataset_name, net_seqLen_name

def evaluate_save_dir_settings(args):
    create_dir(args.base_save_dir)
    args.dataset_save_dir = args.base_save_dir + args.dataset_name + "/"
    create_dir(args.dataset_save_dir)
    args.netSeqlen_save_dir = args.dataset_save_dir + args.net_seqLen_name + "/"
    create_dir(args.netSeqlen_save_dir)
    args.txtRst_save_dir = args.netSeqlen_save_dir + "txtResult/"
    create_dir(args.txtRst_save_dir)
    args.rocRst_save_dir = args.netSeqlen_save_dir + "rocResult/"
    create_dir(args.rocRst_save_dir)
    return args

def parse_args():
    #
    # Setting parameters
    #
    parser = ArgumentParser(description='Implement of pytorch multi frame train')

    #
    # Evaluating parameters
    #
    parser.add_argument("--dataset_dir", default='../dataset/', type=str, help="test_dataset_dir")

    #
    # Evaluate parameters
    #
    parser.add_argument("--thre", default = 0.5, type = float, help="The threshold value")
    parser.add_argument("--is_resize", action = "store_true", help = "whether resize the picture. Default: True")
    parser.add_argument("--base_size", default = 256, type = int, help="The threshold value")
    
    #
    # Save parameters
    #
    parser.add_argument('--base_save_dir', type=str, default='../result/eva_rst/', help='saving dir')

    #
    # Pic Dir evaluated
    #
    parser.add_argument("--rst_dirs", default=None, type=str, nargs='+', help = "rst_dirs")

    #
    # metric whether used
    #
    parser.add_argument("--mIoU", action = "store_false", help = "whether use mIoU metric. Default: True")
    parser.add_argument("--roc_curve", action = "store_false", help = "whether use roc_curve metric. Default: True")
    parser.add_argument("--pd_fa", action = "store_false", help = "whether use pd_fa metric. Default: True")
    parser.add_argument("--my_pd_fa", action = "store_false", help = "whether use my_pd_fa metric. Default: True")
    parser.add_argument("--TPFNFP", action = "store_false", help = "whether use TPFNFP metric. Default: True")

    args = parser.parse_args()

    return args

class Evaluationer(object):
    def __init__(self, args):
        self.args = evaluate_save_dir_settings(args)
        
        ## parameter set
        self.thre = args.thre
        self.base_size = args.base_size

        ## Set dir
        self.total_dataset_dir = args.dataset_dir + args.dataset_name + '/'
        self.overall_mask_dir = self.total_dataset_dir + 'masks/'
        self.txt_dir = self.total_dataset_dir + "video_idx/"
        self.overall_result_dir = args.rst_dir + 'img/'
        
        if not os.path.exists(self.total_dataset_dir):
            raise ValueError("The dataset dir " + self.total_dataset_dir + " does not exit.")
        if not os.path.exists(self.overall_mask_dir):
            raise ValueError("The mask dir " + self.overall_mask_dir + " does not exit.")
        if not os.path.exists(self.txt_dir):
            raise ValueError("The txt dir " + self.txt_dir + " does not exit.")
        if not os.path.exists(self.overall_result_dir):
            raise ValueError("The result dir " + self.overall_result_dir + "does not exit.")
        
        ## evaluation range
        if args.dataset_name == "IRSatVideo-LEO":
            self.txt_name_list = ["test_IRSatVideo-LEO-easy", "test_IRSatVideo-LEO-middle", "test_IRSatVideo-LEO-hard"]
            self.total_txt_name = "test_IRSatVideo-LEO"
            # self.txt_name_list = ["test_IRSatVideo-LEO-try1", "test_IRSatVideo-LEO-try2"]
            # self.total_txt_name = "test_IRSatVideo-LEO-tryTotal"
        elif args.dataset_name == "NUDT-MIRSDT-NEW-v2":
            self.txt_name_list = ["test"]
            self.total_txt_name = ""
            # self.txt_name_list = ["test1", "test2"]
            # self.total_txt_name = " test_total.txt"
        
        ## evaluation
        self.roc = ROCMetric(bins=200)
        self.eval_mIoU = mIoU()
        self.eval_PD_FA = PD_FA()
        self.eval_my_PD_FA = my_PD_FA()
        self.eval_mIoU_P_R_F = SegmentationMetricTPFNFP(nclass=1)
    
    def evaluate(self):
        self.txt_sav = {}
        for txt_name in self.txt_name_list:
            self.evaluate_single(txt_name)
        if args.dataset_name == "IRSatVideo-LEO":
        #if args.dataset_name == "NUDT-MIRSDT-NEW-v2":
            self.evaluate_total()
            print("[Evaluationer evaluate]")

    def evaluate_single(self, txt_name):
        print("------------------------------------------------")
        print("Begin dealing with " + txt_name)
        print("------------------------------------------------")

        txt_path = self.txt_dir + txt_name + '.txt'
        seq_list = []
        with open(txt_path, 'r') as f:
            seq_list = f.read().splitlines()
        if len(seq_list) == 0:
            print(txt_path, " does not have any content.")
            return

        self.roc.reset()
        self.eval_mIoU.reset()
        self.eval_PD_FA.reset()
        self.eval_my_PD_FA.reset()
        self.eval_mIoU_P_R_F.reset()

        for seq in seq_list:
            result_dir = self.overall_result_dir + seq + "/"
            mask_dir = self.overall_mask_dir + seq + "/"
            pic_list = os.listdir(result_dir)
            print("Begin dealing with ", txt_name, "+", seq, ", and the length is ", len(pic_list))
            count = 1
            for pic_name in pic_list:
                count += 1
                if count % 15 == 0:
                    print("Begin dealing with ", txt_name, "+", seq, "+", pic_name)
                pic_path = result_dir + pic_name
                mask_path = mask_dir + pic_name
                rstImg = load_image(pic_path)
                mask = load_image(mask_path)
                rstImg = normalized(rstImg)
                mask = normalized(mask)

                rst_seg = np.zeros(rstImg.shape, dtype=np.float64)
                rst_seg[rstImg > self.thre] = 1.0

                rst_h, rst_w = rst_seg.shape
                mask_h, mask_w = mask.shape

                if args.is_resize or (rst_h != mask_h, rst_w != mask_w):
                    target = cv2.resize(rst_seg, dsize=(self.base_size, self.base_size), interpolation=cv2.INTER_LINEAR)
                    mask = cv2.resize(mask, dsize=(self.base_size, self.base_size), interpolation=cv2.INTER_LINEAR)
                else:
                    target = rst_seg

                #size = target.shape
                H, W = target.shape

                self.roc.update(pred=target, label=mask)
                self.eval_mIoU.update(
                    (torch.from_numpy(target.reshape(1, 1, H, W)) > self.thre),
                    torch.from_numpy(mask.reshape(1, 1, H, W)))
                # self.eval_mIoU.update(
                #     (torch.from_numpy(target.reshape(1, 1, self.base_size, self.base_size)) > self.thre),
                #     torch.from_numpy(mask.reshape(1, 1, self.base_size, self.base_size)))
                #self.eval_PD_FA.update(target, mask, size)
                self.eval_PD_FA.update(target, mask, [H, W])
                self.eval_my_PD_FA.update(target, mask)
                self.eval_mIoU_P_R_F.update(labels=mask, preds=target)

        Yin_pixAcc, Yin_mIoU = self.eval_mIoU.get()
        fpr, tpr, auc = self.roc.get()
        pd, fa = self.eval_PD_FA.get()
        pd_our, fa_our = self.eval_my_PD_FA.get()
        miou_our, prec, recall, fscore = self.eval_mIoU_P_R_F.get()

        print('XinyiYing: pixAcc %.6f, mIoU: %.6f' % (Yin_pixAcc, Yin_mIoU))
        print('AUC: %.6f' % (auc))
        print('Old Pd: %.6f, Old Fa: %.8f, Our Pd: %.6f, Our Fa: %.8f' % (pd, fa, pd_our, fa_our))
        print('Our: mIoU: %.6f, Prec: %.6f, Recall: %.6f, fscore: %.6f' % (miou_our, prec, recall, fscore))
        print("\n")

        ## save index Result
        save_index_path = self.args.txtRst_save_dir + txt_name + '.txt'
        with open(save_index_path, 'w') as f_index:
            f_index.write('XinyiYing: pixAcc %.6f, mIoU: %.6f' % (Yin_pixAcc, Yin_mIoU) + '\n')
            f_index.write('AUC: %.6f' % (auc) + '\n')
            f_index.write('Old Pd: %.6f, Old Fa: %.8f, Our Pd: %.6f, Our Fa: %.8f' % (pd, fa, pd_our, fa_our) + '\n')
            f_index.write(
                'Our: mIoU: %.6f, Prec: %.6f, Recall: %.6f, fscore: %.6f' % (miou_our, prec, recall, fscore) + '\n')

        ## save roc result
        save_roc_path = self.args.rocRst_save_dir + txt_name + '.mat'
        save_dict = {
            'fpr': fpr,
            'tpr': tpr,
            'auc': auc,
            'pd': pd,
            'fa': fa,
            'our_pd': pd_our,
            'our_fa': fa_our,
            'mIoU': Yin_mIoU,
            'out_mIoU': miou_our,
            'prec': prec,
            'recall': recall,
            'fscore': fscore,
        }
        scio.savemat(save_roc_path, save_dict)

        # save key result in each txt
        self.txt_sav[txt_name] = {}

        # fpr, tpr, auc
        fd, ba, td, tn = self.roc.get_all()
        self.txt_sav[txt_name]['roc'] = {
            'fd': fd,
            'ba': ba,
            'td': td,
            'tn': tn
        }

        # mIoU
        total_correct, total_label, total_inter, total_union = self.eval_mIoU.get_all()
        self.txt_sav[txt_name]['mIoU'] = {
            'total_correct': total_correct,
            'total_label': total_label,
            'total_inter': total_inter,
            'total_union': total_union
        }

        # PD FA
        dismatch_pixel, all_pixel, PD, target = self.eval_PD_FA.get_all()
        self.txt_sav[txt_name]['PDFA'] = {
            'dismatch_pixel': dismatch_pixel,
            'all_pixel': all_pixel,
            'PD': PD,
            'target': target
        }

        # my PD FA
        false_detect, background_area, true_detect, target_nums = self.eval_my_PD_FA.get_all()
        self.txt_sav[txt_name]['myPDFA'] = {
            'false_detect': false_detect,
            'background_area': background_area,
            'true_detect': true_detect,
            'target_nums': target_nums
        }

        # mIoU_P_R_F
        total_tp, total_fp, total_fn = self.eval_mIoU_P_R_F.get_all()
        self.txt_sav[txt_name]['mIoU_P_R_F'] = {
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn
        }

    def evaluate_total(self):
        # roc
        fd = self.txt_sav[self.txt_name_list[0]]['roc']['fd']
        ba = self.txt_sav[self.txt_name_list[0]]['roc']['ba']
        td = self.txt_sav[self.txt_name_list[0]]['roc']['td']
        tn = self.txt_sav[self.txt_name_list[0]]['roc']['tn']

        for i in range(len(self.txt_name_list) - 1):
            for j in range(len(fd)):
                fd[j] += self.txt_sav[self.txt_name_list[i + 1]]['roc']['fd'][j]
            ba += self.txt_sav[self.txt_name_list[i + 1]]['roc']['ba']
            for j in range(len(td)):
                td[j] += self.txt_sav[self.txt_name_list[i + 1]]['roc']['td'][j]
            tn += self.txt_sav[self.txt_name_list[i + 1]]['roc']['tn']
        fpr = fd / ba
        tpr = td / tn
        AUC = auc(fpr, tpr)

        # mIoU
        total_correct = self.txt_sav[self.txt_name_list[0]]['mIoU']['total_correct']
        total_label = self.txt_sav[self.txt_name_list[0]]['mIoU']['total_label']
        total_inter = self.txt_sav[self.txt_name_list[0]]['mIoU']['total_inter']
        total_union = self.txt_sav[self.txt_name_list[0]]['mIoU']['total_union']

        for i in range(len(self.txt_name_list) - 1):
            total_correct += self.txt_sav[self.txt_name_list[i + 1]]['mIoU']['total_correct']
            total_label += self.txt_sav[self.txt_name_list[i + 1]]['mIoU']['total_label']
            total_inter += self.txt_sav[self.txt_name_list[i + 1]]['mIoU']['total_inter']
            total_union += self.txt_sav[self.txt_name_list[i + 1]]['mIoU']['total_union']
        Yin_pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
        IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
        Yin_mIoU = IoU.mean()

        # PD FA
        dismatch_pixel = self.txt_sav[self.txt_name_list[0]]['PDFA']['dismatch_pixel']
        all_pixel = self.txt_sav[self.txt_name_list[0]]['PDFA']['all_pixel']
        PD = self.txt_sav[self.txt_name_list[0]]['PDFA']['PD']
        target = self.txt_sav[self.txt_name_list[0]]['PDFA']['target']

        for i in range(len(self.txt_name_list) - 1):
            dismatch_pixel += self.txt_sav[self.txt_name_list[i + 1]]['PDFA']['dismatch_pixel']
            all_pixel += self.txt_sav[self.txt_name_list[i + 1]]['PDFA']['all_pixel']
            PD += self.txt_sav[self.txt_name_list[i + 1]]['PDFA']['PD']
            target += self.txt_sav[self.txt_name_list[i + 1]]['PDFA']['target']
        fa = dismatch_pixel / all_pixel
        pd = PD / target

        # my PD FA
        false_detect = self.txt_sav[self.txt_name_list[0]]['myPDFA']['false_detect']
        background_area = self.txt_sav[self.txt_name_list[0]]['myPDFA']['background_area']
        true_detect = self.txt_sav[self.txt_name_list[0]]['myPDFA']['true_detect']
        target_nums = self.txt_sav[self.txt_name_list[0]]['myPDFA']['target_nums']

        for i in range(len(self.txt_name_list) - 1):
            false_detect += self.txt_sav[self.txt_name_list[i + 1]]['myPDFA']['false_detect']
            background_area += self.txt_sav[self.txt_name_list[i + 1]]['myPDFA']['background_area']
            true_detect += self.txt_sav[self.txt_name_list[i + 1]]['myPDFA']['true_detect']
            target_nums += self.txt_sav[self.txt_name_list[i + 1]]['myPDFA']['target_nums']
        fa_our = false_detect / background_area  #
        pd_our = true_detect / target_nums  #

        # mIoU_P_R_F
        total_tp = self.txt_sav[self.txt_name_list[0]]['mIoU_P_R_F']['total_tp']
        total_fp = self.txt_sav[self.txt_name_list[0]]['mIoU_P_R_F']['total_fp']
        total_fn = self.txt_sav[self.txt_name_list[0]]['mIoU_P_R_F']['total_fn']

        for i in range(len(self.txt_name_list) - 1):
            total_tp += self.txt_sav[self.txt_name_list[i + 1]]['mIoU_P_R_F']['total_tp']
            total_fp += self.txt_sav[self.txt_name_list[i + 1]]['mIoU_P_R_F']['total_fp']
            total_fn += self.txt_sav[self.txt_name_list[i + 1]]['mIoU_P_R_F']['total_fn']
        miou_our = 1.0 * total_tp / (np.spacing(1) + total_tp + total_fp + total_fn)
        prec = 1.0 * total_tp / (np.spacing(1) + total_tp + total_fp)
        recall = 1.0 * total_tp / (np.spacing(1) + total_tp + total_fn)
        fscore = 2.0 * prec * recall / (np.spacing(1) + prec + recall)

        print('XinyiYing: pixAcc %.6f, mIoU: %.6f' % (Yin_pixAcc, Yin_mIoU))
        print('AUC: %.6f' % (AUC))
        print('Old Pd: %.6f, Old Fa: %.8f, Our Pd: %.6f, Our Fa: %.8f' % (pd, fa, pd_our, fa_our))
        print('Our: mIoU: %.6f, Prec: %.6f, Recall: %.6f, fscore: %.6f' % (miou_our, prec, recall, fscore))

        ## save index Result
        save_index_path = self.args.txtRst_save_dir + self.total_txt_name + '.txt'
        with open(save_index_path, 'w') as f_index:
            f_index.write('XinyiYing: pixAcc %.6f, mIoU: %.6f' % (Yin_pixAcc, Yin_mIoU) + '\n')
            f_index.write('AUC: %.6f' % (AUC) + '\n')
            f_index.write('Old Pd: %.6f, Old Fa: %.8f, Our Pd: %.6f, Our Fa: %.8f' % (pd, fa, pd_our, fa_our) + '\n')
            f_index.write(
                'Our: mIoU: %.6f, Prec: %.6f, Recall: %.6f, fscore: %.6f' % (miou_our, prec, recall, fscore) + '\n')

        ## save roc result
        save_roc_path = self.args.rocRst_save_dir + self.total_txt_name + '.mat'
        save_dict = {
            'fpr': fpr,
            'tpr': tpr,
            'auc': AUC,
            'pd': pd,
            'fa': fa,
            'our_pd': pd_our,
            'our_fa': fa_our,
            'mIoU': Yin_mIoU,
            'out_mIoU': miou_our,
            'prec': prec,
            'recall': recall,
            'fscore': fscore,
        }
        scio.savemat(save_roc_path, save_dict)

if __name__ == '__main__':
    args = parse_args()

    rst_dirs = names_standard_format(args.rst_dirs)

    for rst_dir in rst_dirs:
        args.rst_dir = rst_dir
        args.dataset_name, args.net_seqLen_name = get_dataset_name(rst_dir)

        eva = Evaluationer(args)
        eva.evaluate()