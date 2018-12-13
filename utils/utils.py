import functools
import re
import shutil

import numpy as np
import yaml
from torch import nn
from torch import  autograd
import torch
# from visualize import VisdomPlotter
import os
import pdb

from torch.autograd import Variable

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def add_gaussion_noise(x, sigma=0.1, cuda=False):
    noise = Variable(torch.zeros(x.shape)).cuda() if cuda else Variable(torch.zeros(x.shape))
    noise.data.normal_(0, std=sigma)
    return x + noise

def save_checkpoint(state, is_best, filepath='checkpoint.pth.tar'):
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, 'model_best.pth.tar')


def replace_values(yaml_file):
    def _get(dict, list):
        return functools.reduce(lambda d, k: d[k], list, dict)

    def _replace(obj):
        for k, v in obj.items():
            if isinstance(v, dict):
                _replace(v)
            if isinstance(v, str):
                match = re.search(r'{{(.*?)}}', v)
                while match:
                    reference = match.group(1)
                    replace = yaml_file[reference]
                    v = re.sub(r'{{.*?}}', replace, v, count=1)
                    match = re.search(r'{{(.*?)}}', v)
                obj[k] = v

    _replace(yaml_file)
    return yaml_file

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_config(yaml_path='config.yaml'):
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.load(f)
    except:
        print(os.path.abspath(yaml_path))

    config = replace_values(config)
    return config


class minibatch_discriminator(nn.Module):
    def __init__(self, num_channels, B_dim, C_dim):
        super(minibatch_discriminator, self).__init__()
        self.B_dim = B_dim
        self.C_dim =C_dim
        self.num_channels = num_channels
        T_init = torch.randn(num_channels * 4 * 4, B_dim * C_dim) * 0.1
        self.T_tensor = nn.Parameter(T_init, requires_grad=True)

    def forward(self, inp):
        inp = inp.view(-1, self.num_channels * 4 * 4)
        M = inp.mm(self.T_tensor)
        M = M.view(-1, self.B_dim, self.C_dim)

        op1 = M.unsqueeze(3)
        op2 = M.permute(1, 2, 0).unsqueeze(0)

        output = torch.sum(torch.abs(op1 - op2), 2)
        output = torch.sum(torch.exp(-output), 2)
        output = output.view(M.size(0), -1)

        output = torch.cat((inp, output), 1)

        return output


class Utils(object):

    @staticmethod
    def smooth_label(tensor, offset):
        return tensor + offset

    @staticmethod

    # based on:  https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
    def compute_GP(netD, real_data, real_embed, fake_data, LAMBDA):
        BATCH_SIZE = real_data.size(0)
        alpha = torch.rand(BATCH_SIZE, 1)
        alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement() / BATCH_SIZE)).contiguous().view(BATCH_SIZE, 3, 64, 64)
        alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = interpolates.cuda()

        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates, _ = netD(interpolates, real_embed)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

        return gradient_penalty

    @staticmethod
    def save_checkpoint(netD, netG, dir_path, subdir_path, epoch, inverse=False, stage=1):
        path =  os.path.join(dir_path, subdir_path)
        if not os.path.exists(path):
            os.makedirs(path)
        if not inverse:
            torch.save(netD.state_dict(), '{0}/cycle_disc_stage_{2}_epoch_{1}.pth'.format(path, epoch, stage))
            torch.save(netG.state_dict(), '{0}/cycle_gen_stage_{2}_epoch_{1}.pth'.format(path, epoch, stage))
        else:
            torch.save(netD.state_dict(), '{0}/inv_disc_{1}.pth'.format(path, epoch))
            torch.save(netG.state_dict(), '{0}/inv_gen_{1}.pth'.format(path, epoch))

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

