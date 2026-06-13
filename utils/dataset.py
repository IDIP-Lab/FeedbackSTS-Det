import random
import os
import numpy as np

import torch
from torch.utils.data.dataset import Dataset

from .utils import load_image, normalized, random_crop_seq, batch_resize, get_img_norm_cfg

IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF', '.mat')

__all__ = ['TrainSetIRSatVideoLoader']

def split_string_every_n_chars(s, n):
    return [s[i:i+n] for i in range(0, len(s), n)]

class TrainSetIRSatVideoLoader(Dataset):
    def __init__(self, dataset_dir, dataset_name,  seq_len, patch_size, 
                 sample_rate=1, pos_prob=0.5, img_norm_cfg = None):
        super(TrainSetIRSatVideoLoader).__init__()
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.seq_len = seq_len
        self.sample_rate = sample_rate
        self.patch_size = patch_size
        self.pos_prob = pos_prob

        self.train_list = []
        with open(self.dataset_dir + dataset_name + '/video_idx/train_' + dataset_name + '.txt', 'r') as f:
        # with open(self.dataset_dir + dataset_name + '/video_idx/trainTry_' + dataset_name + '.txt', 'r') as f:
            self.train_list = f.read().splitlines()
        seq_list = []
        for seq_dir in self.train_list:
            with open(dataset_dir + dataset_name + '/img_idx/' + seq_dir + '.txt', 'r') as f:
                img_list = f.read().splitlines()
                if(len(img_list) == 1):
                    img_list = split_string_every_n_chars(img_list[0], 4)
            seq_list = seq_list + [seq_dir for _ in range(len(img_list))]
        self.total_len = len(seq_list)
        self.seq_list = seq_list

        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        self.tranform = augumentation()
    
    def __getitem__(self, idx):
        seq_name = random.sample(self.seq_list, 1)[0]
        with open(self.dataset_dir + self.dataset_name + '/img_idx/' + seq_name + '.txt', 'r') as f:
            img_list = f.read().splitlines()
            if(len(img_list) == 1):
                img_list = split_string_every_n_chars(img_list[0], 4)
        img_ext = os.path.splitext(os.listdir(self.dataset_dir + self.dataset_name + '/images/' + seq_name)[0])[-1]
        if not img_ext in IMG_EXTENSIONS:
            raise TypeError("Unrecognized image extensions.")    

        img_seq = []
        mask_seq = []

        start_index = random.randint(0, len(img_list)-1)
        for i in range(0, self.seq_len):
            cur_idx = start_index + i
            if cur_idx > len(img_list) - 1:
                cur_idx = len(img_list) - 1
            img = load_image(self.dataset_dir + self.dataset_name + '/images/' + seq_name + '/' + img_list[cur_idx] + img_ext)
            img = normalized(img.astype(np.float32), img_norm_cfg = self.img_norm_cfg)
            mask = load_image(self.dataset_dir + self.dataset_name + '/masks/' + seq_name + '/' + img_list[cur_idx] + img_ext)
            mask = normalized(mask.astype(np.float32))

            img_seq.append(img)
            mask_seq.append(mask)
            
        img_seq = np.stack(img_seq, 0)
        mask_seq = np.stack(mask_seq, 0)
        img_patch, mask_patch = random_crop_seq(img_seq, mask_seq, self.patch_size, pos_prob=self.pos_prob)
        img_patch, mask_patch = self.tranform(img_patch, mask_patch)
        img_patch, mask_patch = img_patch[np.newaxis, :, :, :], mask_patch[np.newaxis, :, :, :]
        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))

        return img_patch, mask_patch
    
    def __len__(self):
        return self.total_len // self.sample_rate
    
class IRSatVideoLoader(Dataset):
    def __init__(self, dataset_dir, dataset_name,  seq_len, sample_space = 1,
                 patch_size = 256, pos_prob=0.5, img_norm_cfg = None, mode = "train"):
        super(IRSatVideoLoader).__init__()
        
        assert seq_len > 1, "Variable seq_len should be larger than 1"
        assert sample_space > 0, "Variable seq_space should be larger than 0"
        assert sample_space <= seq_len, "Variable seq_space should be not larger than seq_len."
        
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.seq_len = seq_len
        if mode == "train":
            self.sample_space = sample_space
        else:
            self.sample_space = seq_len
        self.patch_size = patch_size
        self.pos_prob = pos_prob
        self.mode = mode

        self.total_img_dir = dataset_dir + dataset_name + '/images/'
        self.total_mask_dir = dataset_dir + dataset_name + '/masks/'

        self.seq_list = []
        txt_path = self.dataset_dir + dataset_name + '/video_idx/' + mode + "_" + dataset_name + '.txt'
        with open(txt_path, 'r') as f:
            self.seq_list = f.read().splitlines()
        
        name_list = []
        self.seq_path_list = []
        for seq_dir in self.seq_list:
            with open(dataset_dir + dataset_name + '/img_idx/' + seq_dir + '.txt', 'r') as f:
                name_list = f.read().splitlines()
            if(len(name_list) == 1):
                name_list = split_string_every_n_chars(name_list[0], 4)
            
            length = len(name_list)
            start_index = 0
            for i in range(0, length , self.sample_space):
                start_index = i
                if (i + seq_len) > length:
                    start_index = length - seq_len
                end_index = i + seq_len

                self.seq_path_list.append([seq_dir + "/" + name + ".png" 
                                          for name in name_list[start_index : end_index]])

        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        self.tranform = augumentation()
    
    def __getitem__(self, idx):
        seq_path_patch = self.seq_path_list[idx]

        img_seq = []
        mask_seq = []

        for seq_path in seq_path_patch:
            img_path = self.total_img_dir + seq_path
            mask_path = self.total_mask_dir + seq_path
            img = load_image(img_path)
            img = normalized(img.astype(np.float32), self.img_norm_cfg)
            mask = load_image(mask_path)
            mask = normalized(mask.astype(np.float32))
            img_seq.append(img)
            mask_seq.append(mask)

        img_seq = np.stack(img_seq, 0)
        mask_seq = np.stack(mask_seq, 0)
        if self.mode == "train":
            img_patch, mask_patch = random_crop_seq(img_seq, mask_seq, self.patch_size, pos_prob=self.pos_prob)
            img_patch, mask_patch = self.tranform(img_patch, mask_patch)
        else:
           img_patch, mask_patch = img_seq, mask_seq 
        img_patch, mask_patch = img_patch[np.newaxis, :, :, :], mask_patch[np.newaxis, :, :, :]
        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))
        
        if self.mode == "train":
            return img_patch, mask_patch
        else:
            shape = img_seq.shape
            H, W = shape[-2], shape[-1]
            return img_patch, mask_patch, seq_path_patch, H, W
    
    def __len__(self):
        return len(self.seq_path_list)
    
class MIRSDTLoader(Dataset):
    def __init__(self, dataset_dir, dataset_name,  seq_len, sample_space = 1,
                 patch_size = 256, img_norm_cfg = None, mode = "train"):
        super(MIRSDTLoader).__init__()
        
        assert seq_len > 1, "Variable seq_len should be larger than 1"
        assert sample_space > 0, "Variable seq_space should be larger than 0"
        assert sample_space <= seq_len, "Variable seq_space should be not larger than seq_len."
        
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.seq_len = seq_len
        if mode == "train":
            self.sample_space = sample_space
        else:
            self.sample_space = seq_len
        self.patch_size = patch_size
        self.mode = mode

        self.total_img_dir = dataset_dir + dataset_name + '/images/'
        self.total_mask_dir = dataset_dir + dataset_name + '/masks/'

        self.seq_list = []
        txt_path = self.dataset_dir + dataset_name + '/video_idx/' + mode + '.txt'
        with open(txt_path, 'r') as f:
            self.seq_list = f.read().splitlines()
        
        name_list = []
        self.seq_path_list = []
        for seq_dir in self.seq_list:
            with open(dataset_dir + dataset_name + '/img_idx/' + seq_dir + '.txt', 'r') as f:
                name_list = f.read().splitlines()
            if(len(name_list) == 1):
                name_list = split_string_every_n_chars(name_list[0], 4)
            
            length = len(name_list)
            start_index = 0
            for i in range(0, length , self.sample_space):
                start_index = i
                if (i + seq_len) > length:
                    start_index = length - seq_len
                end_index = i + seq_len

                self.seq_path_list.append([seq_dir + "/" + name + ".png" 
                                          for name in name_list[start_index : end_index]])

        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        self.tranform = augumentation()
    
    def __getitem__(self, idx):
        seq_path_patch = self.seq_path_list[idx]

        img_seq = []
        mask_seq = []

        for seq_path in seq_path_patch:
            img_path = self.total_img_dir + seq_path
            mask_path = self.total_mask_dir + seq_path
            img = load_image(img_path)
            img = normalized(img.astype(np.float32), self.img_norm_cfg)
            mask = load_image(mask_path)
            mask = normalized(mask.astype(np.float32))
            img_seq.append(img)
            mask_seq.append(mask)

        img_seq = np.stack(img_seq, 0)
        mask_seq = np.stack(mask_seq, 0)
        img_patch, mask_patch = batch_resize(img_seq, mask_seq, self.patch_size, self.patch_size)
        if self.mode == "train":
            img_patch, mask_patch = self.tranform(img_patch, mask_patch)
        img_patch, mask_patch = img_patch[np.newaxis, :, :, :], mask_patch[np.newaxis, :, :, :]
        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))
        
        if self.mode == "train":
            return img_patch, mask_patch
        else:
            shape = img_seq.shape
            H, W = shape[-2], shape[-1]
            return img_patch, mask_patch, seq_path_patch, H, W
    
    def __len__(self):
        return len(self.seq_path_list)
    
class augumentation(object):
    def __call__(self, input, target):
        if random.random()<0.5:
            input = input[:, ::-1, :]
            target = target[:, ::-1, :]
        if random.random()<0.5:
            input = input[:, :, ::-1]
            target = target[:, :, ::-1]
        if random.random()<0.5:
            input = input[::-1, :, :]
            target = target[::-1, :, :]
        if random.random()<0.5:
            input = input.transpose(0, 2, 1)
            target = target.transpose(0, 2, 1)
        return input, target
        