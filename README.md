# Symmetric Learning (SL) via Symmetric Cross Entropy (SCE) loss
Code for ICCV2019 "Symmetric Cross Entropy for Robust Learning with Noisy Labels" https://arxiv.org/abs/1908.06112

## Requirements
- Python 3.5.2
- Tensorflow 1.10.1 
- Keras 2.2.2

## Usage
Simply run the code by python3 train_models.py

It can config with dataset, model, epoch, batchsize, noise_rate, symmetric or asymmetric type noise

## The Pytorch reimplementation
The Pytorch version is implemented by Hanxun Huang. The code can be found here: https://github.com/HanxunHuangLemonBear/SCELoss-Reproduce

## Citing this work
If you use this code in your work, please cite the accompanying paper:

```
@inproceedings{wang2019symmetric,
  title={Symmetric cross entropy for robust learning with noisy labels},
  author={Wang, Yisen and Ma, Xingjun and Chen, Zaiyi and Luo, Yuan and Yi, Jinfeng and Bailey, James},
  booktitle={IEEE International Conference on Computer Vision},
  year={2019}
}
```
