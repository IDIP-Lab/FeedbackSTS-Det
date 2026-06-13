import torch
import cv2
import numpy as np
import random
import time
import os

def create_dir(dir):
    if not os.path.exists(os.path.dirname(dir)):
        os.makedirs(os.path.dirname(dir))

def get_optimizer(net, optimizer_name, scheduler_name, optimizer_settings, scheduler_settings):
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=optimizer_settings['lr'])
    elif optimizer_name == 'Adagrad':
        optimizer  = torch.optim.Adagrad(net.parameters(), lr=optimizer_settings['lr'])
    elif optimizer_name == 'SGD':
        optimizer  = torch.optim.SGD(net.parameters(), lr=optimizer_settings['lr'])
    
    if scheduler_name == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_settings['step'], gamma=scheduler_settings['gamma'])
    elif scheduler_name   == 'CosineAnnealingLR':
        scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_settings['epochs'], eta_min=scheduler_settings['min_lr'])
    
    return optimizer, scheduler

def load_image(srcpath):
    img=cv2.imdecode(np.fromfile(srcpath, dtype=np.uint8), -1)

    if hasattr(img, 'ndim')==False:
        print("Failed to read " + srcpath)

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return img
    # return img.astype(np.float32)

def save_image(img, save_path):
    _, img_encoded = cv2.imencode('.png', img)
    img_encoded.tofile(save_path)

def normalized(input_img, img_norm_cfg = None):
    img = None
    if (img_norm_cfg):
        img = (input_img - img_norm_cfg["mean"]) / (img_norm_cfg["std"])
    else:
        img = input_img

    max_val = np.max(img)
    min_val = np.min(img)
    # if max_val > 0:
    #     img = (img - min_val) / (max_val - min_val)
    if max_val == min_val:
        img = np.zeros_like(img, dtype=np.float32)
    else:
        img = (img - min_val) / (max_val - min_val)
    return img

def random_crop_seq(img_seq, mask_seq, patch_size, pos_prob=False): 
    _, h, w = img_seq.shape
    if min(h, w) < patch_size:
        for i in range(len(img_seq)):
            img_seq[i,:,:] = np.pad(img_seq[i,:,:], ((0, 0),(0, max(h, patch_size)-h),(0, max(w, patch_size)-w)), mode='constant')
            mask_seq[i,:,:] = np.pad(mask_seq[i,:,:], ((0, 0),(0, max(h, patch_size)-h),(0, max(w, patch_size)-w)), mode='constant')
            _, h, w = img_seq.shape
    
    cur_prob = random.random()
    
    if pos_prob == None or cur_prob > pos_prob or mask_seq.max() == 0:
        h_start = random.randint(0, h - patch_size)
        w_start = random.randint(0, w - patch_size)
    else:
        loc = np.where(mask_seq > 0)
        if len(loc[0]) <= 1:
            idx = 0
        else:
            idx = random.randint(0, len(loc[0])-1)
        h_start = random.randint(max(0, loc[1][idx] - patch_size), min(loc[1][idx], h-patch_size))
        w_start = random.randint(max(0, loc[2][idx] - patch_size), min(loc[2][idx], w-patch_size))
        
    h_end = h_start + patch_size
    w_end = w_start + patch_size
    img_patch_seq = img_seq[:, h_start:h_end, w_start:w_end]
    mask_patch_seq = mask_seq[:, h_start:h_end, w_start:w_end]

    return img_patch_seq, mask_patch_seq

def batch_resize(img_seq, mask_seq, resize_h, resize_w):
    frame_num, _, _ = img_seq.shape
    img_patch_seq = np.zeros((frame_num, resize_h, resize_w), dtype = img_seq.dtype)
    mask_patch_seq = np.zeros((frame_num, resize_h, resize_w), dtype = mask_seq.dtype)

    for i in range(frame_num):
        img_patch_seq[i, :, :] = cv2.resize(img_seq[i, :, :], dsize=(resize_h, resize_w), 
                                            interpolation = cv2.INTER_LINEAR)
        mask_patch_seq[i, :, :] = cv2.resize(mask_seq[i, :, :], dsize=(resize_h, resize_w), 
                                            interpolation = cv2.INTER_NEAREST)
    return img_patch_seq, mask_patch_seq


def save_checkpoint(state, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(state, save_path)
    return save_path

def get_img_norm_cfg(dataset_name, dataset_dir):
    if dataset_name == 'IRSatVideo-LEO':   
        img_norm_cfg = {'mean': 72.1040267944336, 'std': 12.302865028381348}
    elif "MIRSDT" in dataset_name:
        img_norm_cfg = {'mean': 105.4025, 'std': 26.6452}
    else:
        with open(dataset_dir+'/'+dataset_name+'/img_idx/train_' + dataset_name + '.txt', 'r') as f:
            train_list = f.read().splitlines()
        with open(dataset_dir+'/'+dataset_name+'/img_idx/test_' + dataset_name + '.txt', 'r') as f:
            test_list = f.read().splitlines()
        img_list = train_list + test_list
        img_dir = dataset_dir + '/' + dataset_name + '/images/'
        mean_list = []
        std_list = []
        for img_pth in img_list:
            try:
                img = load_image((img_dir + img_pth).replace('//','/')+'.jpg')
            except:
                try:
                    img = load_image((img_dir + img_pth).replace('//','/')+'.png')
                except:
                    img = load_image((img_dir + img_pth).replace('//','/')+'.bmp')
            img = np.array(img, dtype=np.float32)
            mean_list.append(img.mean())
            std_list.append(img.std())
        img_norm_cfg = dict(mean=float(np.array(mean_list).mean()), std=float(np.array(std_list).mean()))
        print(dataset_name)
        print(img_norm_cfg)
    return img_norm_cfg

def train_save_dir_settings(args):
    args.time_name = time.strftime('%Y%m%dT%H-%M-%S', time.localtime(time.time()))
    args.dir_name = '{}_{}'.format(args.dataset_name, args.time_name) + '/'
    create_dir(args.base_dir)
    log_dir = args.base_dir + 'log/'
    create_dir(log_dir)
    args.save_log_dir = log_dir + args.dir_name

    pth_dir = args.base_dir + 'parameters/'
    create_dir(pth_dir)
    args.pth_save_dir = pth_dir

    args.dataset_save_dir = args.pth_save_dir + args.dataset_name + '/'
    create_dir(args.dataset_save_dir)
    return args

def test_save_dir_settings(args):
    create_dir(args.base_save_dir)
    args.dataset_save_dir = args.base_save_dir + args.dataset_name + "/"
    create_dir(args.dataset_save_dir)
    return args