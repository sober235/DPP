from tracemalloc import take_snapshot
import wave
import torch
import torch.nn as nn
import scipy.io as sio
import numpy as np
import torch.utils.data as Data
import time
import os
from os.path import join
import torch.nn.functional as F
from argparse import ArgumentParser
from torch.autograd import Variable
from data_proc import  getTestingData_Wave_com
from net import CovDecoder
from utils_new import fftyc,ifftyc,r2c, c2r
import torch.autograd as autograd
import itertools
import torch.nn.functional as F
import torchvision.transforms as T
parser = ArgumentParser(description='dip')
name = 'dip'

parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed = 1337

wave_vcc = 1

org, atb, mask, filt, psf = getTestingData_Wave_com(nImg=1, wave=wave, vcc=vcc, wave_vcc=wave_vcc, device=device)

org = torch.from_numpy(org)
atb = torch.from_numpy(atb)
mask = torch.from_numpy(mask)
filt = torch.from_numpy(filt)
psf = torch.from_numpy(psf)


atb = c2r(atb) 
nb, nc, nx, ny = org.shape
print(atb.shape)

class Mydata(Data.Dataset):
    def __init__(self):
        super(Mydata,self).__init__()
        self.org = org
        self.atb = atb
        self.mask = mask
        self.filt = filt
        self.psf = psf
 
    def __getitem__(self,index):
        orgk = self.org[index]
        atbk = self.atb[index]
        mask = self.mask[index]
        psf = self.psf[index]
        filtk = self.filt[index]
        return orgk, atbk, mask, filtk, psf
    def __len__(self):
        return len(self.org)

batch_size = 1
lr_G = 0.0001 # gaussian random
gain = 0.05  # gaussian random
layerNo = 5

criterion = nn.MSELoss()
out = torch.zeros([nb, nc, nx, ny]).to(device)
trainset = Mydata()
train_loader = Data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

netG = CovDecoder(nch_in=2,  nch_h=256, nch_out = 2,  layerNo=10, filter_size=3, in_size=[3,3], out_size=[nx,ny]).to(device)
netK = CovDecoder(nch_in=2,  nch_h=64,  nch_out = 48, layerNo=5,  filter_size=3, in_size=[3,3], out_size=[11,11]).to(device) # csm1
netK2 = CovDecoder(nch_in=2, nch_h=64,  nch_out = 48, layerNo=5,  filter_size=3, in_size=[3,3], out_size=[11,11]).to(device) # csm2
netP = CovDecoder(nch_in=2,  nch_h=64,  nch_out = 2,  layerNo=5,  filter_size=3, in_size=[3,3], out_size=[11,11]).to(device) # 背景相位
init_weights(netG, init_type='normal', init_gain=1e-2)
init_weights(netK, init_type='normal', init_gain=1e-6)
init_weights(netK2, init_type='normal', init_gain=1e-6)
init_weights(netP, init_type='normal', init_gain=1e-6)
paramsG = netG.parameters()
paramsK = netK.parameters()
paramsK2 = netK2.parameters()
paramsP = netP.parameters()
params = itertools.chain(paramsG,paramsK) # 参数联合
params = itertools.chain(params,paramsK2)
params = itertools.chain(params,paramsP)
ncoil = 48
# end I-domain

optim = torch.optim.Adam(params, lr=lr_G, amsgrad=True)

eta = torch.randn([1,2,3,3]).to(device)
zeta = torch.randn([1,2,3,3]).to(device)
phi_size = 5
kernel_size = 5

tv_cof_wave_vcc = 0.000268555
tv_cof_wave = 0.000505

mask_type = 'vd_4x_acs6'
acc = 4
num_epoch = 2000

def FCNN(x,kernel,pad):
    conv_r,conv_i = torch.chunk(kernel,2,1)
    conv_r = torch.permute(conv_r,[1,0,2,3])
    conv_i = torch.permute(conv_i,[1,0,2,3])
    y_r,y_i = torch.chunk(x,2,1)
    x_r = F.conv2d(y_r,conv_r,padding=pad) - F.conv2d(y_i,conv_i,padding=pad)
    x_i = F.conv2d(y_i,conv_r,padding=pad) + F.conv2d(y_r,conv_i,padding=pad)
    out = torch.cat([x_r,x_i],1)
    return out

zpad = T.CenterCrop((256, 250)) # 这里控制着输入数据的维度

for i, (org,atb,mask,filt,psf) in enumerate(train_loader):
    org,atb,mask,filt,psf = org.type(torch.FloatTensor).to(device), atb.type(torch.FloatTensor).to(device),\
            mask.to(device),filt.to(device), psf.to(device)      
    T = 0
    for epoch in range(num_epoch):
        # k-space
        optim.zero_grad()
        t_start = time.time()
        ksp_H = netG(eta)
        phi = netP(zeta)  
        ksp = FCNN(ksp_H, phi, phi_size) 
        ksp = 0.2*ksp + 0.8*c2r(torch.conj(torch.flip(torch.flip(r2c(ksp_H), [2]), [3])))
        kernel = netK(zeta) 
        kernel_H = netK2(zeta) 
        ksp = FCNN(ksp, kernel, kernel_size) 
        ksp_H = FCNN(ksp_H, kernel_H, kernel_size) 
        ksp = c2r(torch.cat([r2c(ksp), r2c(ksp_H)],1)) 
        kspw = r2c(ksp)
        kspw = ifftyc(kspw)
        kspw = kspw * psf
        kspw = fftyc(kspw)
        kspw = c2r(kspw)
        loss = criterion(c2r(r2c(kspw)*mask*filt), c2r(r2c(atb)*mask*filt))
        loss.backward()
        optim.step()
        t_end = time.time()
        T = T + t_end - t_start
        print('epoch %d: Loss: %.4f' %(epoch, loss.item()))
      
out_time = "time: %.6f_avg: %.6f\n" %(T, T)
print(out_time)

ncoil = ncoil

out = ksp.cpu().data.numpy()

out = out[0, :ncoil, :, :] + 1j * out[0, ncoil:, :, :]
out = np.transpose(out, [1,2,0])
if wave_vcc:
    result_path = "/data/data42/LiuCongcong/Cui_Zhuoxu/PF_bayes/results_est_phase/"
    sio.savemat(join(result_path, 'layer%dmask%s_256x250_acc%d_epoch%d_phase_size%d.mat'%(layerNo,mask_type,acc,num_epoch,phi_size)), {'recon': out})
elif wave:
    result_path = "/data/data42/LiuCongcong/Cui_Zhuoxu/PF_bayes/results_est/wave"
    sio.savemat(join(result_path, 'mask_type_%s_256x250_acc%dx_acs0_num_epoch%d.mat'%(mask_type,acc,num_epoch)), {'recon': out})

