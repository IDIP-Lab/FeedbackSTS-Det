# FeedbackSTS-Det:Sparse Frames-Based Spatio-Temporal Semantic Feedback Network for Moving Infrared Small Target Detection

 Official implementation of paper "FeedbackSTS-Det:Sparse Frames-Based Spatio-Temporal Semantic Feedback Network for Moving Infrared Small Target Detection".

## Network Structure

![Overall Framework](./pic/Overall_framework.png)
*Figure 1: Overall framework of FeedbackSTS-Det*

![BFBM](./pic/BFBM.png)
*Figure 2: Detailed architecture of Basic feedback module (BFBM)*

## Requirements

* **Python 3.8**
* **Windows10, Ubuntu18.04 or higher**
* **NVDIA GeForce RTX 4080**
* **Pytorch 2.4.1**
* **CUDA 11.8**
* **More details from requirements.txt**

## Compilation

Before running the code, the `DCN` module must be compiled. Execute the following commands:
```bash
cd model/dcn
sh make.sh
```

## Dataset

We used IRSatVideo-LEO and NUDT-MIRSDT for training. The two datasets could be found and downloaded in: [IRSatVideo-LEO](https://github.com/XinyiYing/RFR) and [NUDT-MIRSDT](https://github.com/TinaLRJ/Multi-frame-infrared-small-target-detection-DTUM).

For the NUDT-MIRSDT dataset, we have prepared a pre-processed version adapted to our code format. It can be downloaded in [Google Cloud](https://drive.google.com/file/d/1afTlcifxpYJJ1e2r9BJyooL0D3ImNBO7/view?usp=drive_link) or [Baidu Cloud]( https://pan.baidu.com/s/1ZH2hdVauW4Zy5_1noCbuKA?pwd=ggkp) with code of "ggkp".

下载后，数据集构筑形式如下：


## Commands
### Commands for Training
#### Single GPU Training
* **Train on IRSatVideo-LEO dataset**
```bash
python train.py \
  --dataset_dir <Your own dataset directory> \
  --dataset_names IRSatVideo-LEO \
  --seq_len 5 \
  --sample_space 3 \
  --batchSize 6 \
  --patchSize 256
```
* **Train on NUDT-MIRSDT dataset**
```bash
python train.py \
  --dataset_dir <Your own dataset directory> \
  --dataset_names NUDT-MIRSDT-NEW-v2 \
  --seq_len 5 \
  --sample_space 1 \
  --batchSize 6 \
  --patchSize 256
```
* **Specify a GPU Device**
When running on a server with multiple GPUs, you can specify a particular GPU device by adding the `--gpu <GPU_ID>` argument. For example, 
```bash
python train.py \
  --dataset_dir <Your own dataset directory> \
  --dataset_names IRSatVideo-LEO \
  --seq_len 5 \
  --sample_space 3 \
  --batchSize 6 \
  --patchSize 256 \
  --gpu 2   # Use GPU 2
```
#### Multiple GPU Training

### Commands for Testing

### Commands for Evaluation

# Citation

```Citation
@misc{huang2026feedbacksts,
      title={FeedbackSTS-Det: Sparse Frames-Based Spatio-Temporal Semantic Feedback Network for Moving Infrared Small Target Detection},
      author={Huang, Yian and Qin, Qing and Mao, Aji and Qiu, Xiangyu and Xu, Liang and Zhang, Xian and Peng, Zhenming},
      year={2026},
      eprint={2601.14690},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.14690},
}
```
