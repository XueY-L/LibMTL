import random
import torch
import torch.nn as nn
import copy
import torchvision.models as tmodels
import numpy as np


def load_model(domain_ls:list):
    model_path = {  # pretrained resnet50-lr0.005
        'clipart':'./ckpt_res50/checkpoint/domainnet_domainbed_hparam/ckpt_clipart__sgd_lr-s0.005_lr-w0.0005_bs32_seed42_source-[]_DomainNet_resnet50-1.0x_singletraining-domainbedhparam_lrd-[-2, -1]_wd-0.0005.pth',
        'painting':'./ckpt_res50/checkpoint/domainnet_domainbed_hparam/ckpt_painting__sgd_lr-s0.005_lr-w0.0005_bs32_seed42_source-[]_DomainNet_resnet50-1.0x_singletraining-domainbedhparam_lrd-[-2, -1]_wd-0.0005.pth',
        'real':'./ckpt_res50/checkpoint/domainnet_domainbed_hparam/ckpt_real__sgd_lr-s0.005_lr-w0.0005_bs32_seed42_source-[]_DomainNet_resnet50-1.0x_singletraining-domainbedhparam_lrd-[-2, -1]_wd-0.0005.pth',
        'sketch':'./ckpt_res50/checkpoint/domainnet_domainbed_hparam/ckpt_sketch__sgd_lr-s0.005_lr-w0.0005_bs32_seed42_source-[]_DomainNet_resnet50-1.0x_singletraining-domainbedhparam_lrd-[-2, -1]_wd-0.0005.pth',
        'infograph':'./ckpt_res50/checkpoint/domainnet_domainbed_hparam/ckpt_infograph__sgd_lr-s0.005_lr-w0.0005_bs32_seed42_source-[]_DomainNet_resnet50-1.0x_singletraining-domainbedhparam_lrd-[-2, -1]_wd-0.0005.pth',
        'quickdraw':'./ckpt_res50/checkpoint/domainnet_domainbed_hparam/ckpt_quickdraw__sgd_lr-s0.005_lr-w0.0005_bs32_seed42_source-[]_DomainNet_resnet50-1.0x_singletraining-domainbedhparam_lrd-[-2, -1]_wd-0.0005.pth',
    }

    # model_path = {  # pretrained resnet50-lr0.001
    #     'clipart':'./ckpt_res50/checkpoint/domainnet_domainbed_hparam/hparam_lr/ckpt_clipart__sgd_lr-s0.001_lr-w0.0005_bs32_seed42_source-[]_DomainNet_resnet50-1.0x_singletraining-domainbedhparam2_lrd-[-2, -1]_wd-0.0005.pth',
    #     'painting':'./ckpt_res50/checkpoint/domainnet_domainbed_hparam/hparam_lr/ckpt_painting__sgd_lr-s0.001_lr-w0.0005_bs32_seed42_source-[]_DomainNet_resnet50-1.0x_singletraining-domainbedhparam_lrd-[-2, -1]_wd-0.0005.pth',
    #     'real':'./ckpt_res50/checkpoint/domainnet_domainbed_hparam/hparam_lr/ckpt_real__sgd_lr-s0.001_lr-w0.0005_bs32_seed42_source-[]_DomainNet_resnet50-1.0x_singletraining-domainbedhparam_lrd-[-2, -1]_wd-0.0005.pth',
    #     'sketch':'./ckpt_res50/checkpoint/domainnet_domainbed_hparam/hparam_lr/ckpt_sketch__sgd_lr-s0.001_lr-w0.0005_bs32_seed42_source-[]_DomainNet_resnet50-1.0x_singletraining-domainbedhparam2_lrd-[-2, -1]_wd-0.0005.pth',
    #     'infograph':'./ckpt_res50/checkpoint/domainnet_domainbed_hparam/hparam_lr/ckpt_infograph__sgd_lr-s0.001_lr-w0.0005_bs32_seed42_source-[]_DomainNet_resnet50-1.0x_singletraining-domainbedhparam_lrd-[-2, -1]_wd-0.0005.pth',
    #     'quickdraw':'./ckpt_res50/checkpoint/domainnet_domainbed_hparam/hparam_lr/ckpt_quickdraw__sgd_lr-s0.001_lr-w0.0005_bs32_seed42_source-[]_DomainNet_resnet50-1.0x_singletraining-domainbedhparam_lrd-[-2, -1]_wd-0.0005.pth',
    # }

    param_ls = []
    for d in domain_ls:
        print(f'==> Loading {d} Model')
        param_d = torch.load(model_path[d])['net']
        param_ls.append(param_d)
    return param_ls