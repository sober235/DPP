from json import load
import time
import numpy as np
from numpy.core.fromnumeric import transpose
import scipy.io as scio
import h5py
import sigpy.mri as mr
import mat73
from utils_np import r2c,c2r,Emat_xyt,fft2c
from generate_mask import get_cartesian_mask
import torch

def FFT2c(x):
    nb, nc, nx, ny = np.shape(x)

    x = np.fft.ifftshift(x, axes=2)
    x = np.transpose(x,[0,1,3,2])
    x = np.fft.fft(x, axis=-1)
    x = np.transpose(x,[0,1,3,2])
    x = np.fft.fftshift(x, axes=2)/np.math.sqrt(nx)
    x = np.fft.ifftshift(x, axes=3)
    x = np.fft.fft(x, axis=-1)
    x = np.fft.fftshift(x, axes=3)/np.math.sqrt(ny)
    return x

def IFFT2c(x):
    nb, nc, nx, ny = np.shape(x)
    x = np.fft.ifftshift(x, axes=2)
    x = np.transpose(x,[0,1,3,2])
    x = np.fft.ifft(x, axis=-1)
    x = np.transpose(x,[0,1,3,2])
    x = np.fft.fftshift(x, axes=2)*np.math.sqrt(nx)
    x = np.fft.ifftshift(x, axes=3)
    x = np.fft.ifft(x, axis=-1)
    x = np.fft.fftshift(x, axes=3)*np.math.sqrt(ny)
    return x

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)

def crop(img,cropx,cropy):
    nb,c,y,x = img.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2
    return img[:,:,starty:starty+cropy, startx:startx+cropx]



def getTestingData_Wave_com(nImg=1, wave=1, vcc=0, wave_vcc=0, size=[256, 250], device=0): # k-space
    tic()
    A = mat73.loadmat('/data/data42/LiuCongcong/Qiu_Matlab_Project/LORAKS_v2/examples/Jia_Reduced_Fov/DATA.mat') 
    org = A['DATA']
    org = np.transpose(org, [2, 0, 1]) 
    org = np.expand_dims(org, 0)  
    org = np.concatenate([org, np.conjugate(np.flip(np.flip(org, 2), 3))], 1)

    B = scio.loadmat('/data/data42/LiuCongcong/Qiu_Matlab_Project/LORAKS_v2/examples/Jia_Reduced_Fov/mask_vd_256x250_acs6.mat')
    mask = B['mask'][:]
    mask = mask.astype(np.complex64)
    mask = np.tile(mask, [nImg, 24, 1, 1])
    mask = np.concatenate([mask, (np.flip(np.flip(mask, 2), 3))], 1)

    C = scio.loadmat('/data/data42/LiuCongcong/Qiu_Matlab_Project/LORAKS_v2/examples/Jia_Reduced_Fov/filt_256x250_3e-2_0_2.mat')
    h = C['weight'][:]
    h = h.astype(np.complex64)
    h = np.tile(h, [nImg, 1, 1])
    h = np.expand_dims(h, 0)

    D = mat73.loadmat('/data/data42/LiuCongcong/Qiu_Matlab_Project/LORAKS_v2/examples/Jia_Reduced_Fov/PsfY_crop.mat')
    f = D['PsfY'][:]
    f = np.transpose(f, [2, 0, 1])
    f = np.expand_dims(f, 0)
    f = f.astype(np.complex64)
    f = np.concatenate([f, np.conjugate(np.flip(f, 2))], 1)

    toc()
    print('Undersampling')
    tic()
    org = np.fft.ifftshift(org, 3)
    org = np.fft.ifft(org)
    org = np.fft.fftshift(org, 3) * np.math.sqrt(size[1]) 
    org = org * f
    org = np.fft.ifftshift(org, 3)
    org = np.fft.fft(org)
    org = np.fft.fftshift(org, 3) / np.math.sqrt(size[1]) 

    orgk, atb, minv = generateUndersampled(org, mask)

    toc()
    print('Data prepared!')
    return orgk, atb, mask, h, f
    

def usp(x,mask,nch,nrow,ncol):
    """ This is a the A operator as defined in the paper"""
    kspace=np.reshape(x,(nch,nrow,ncol))
    res=kspace[mask!=0]
    return kspace,res

def usph(kspaceUnder,mask,nch,nrow,ncol):
    """ This is a the A^T operator as defined in the paper"""
    temp=np.zeros((nch,nrow,ncol),dtype=np.complex64)
    temp[mask!=0]=kspaceUnder
    minv=np.std(temp)
    temp=temp/minv
    return temp,minv

def generateUndersampled(org,mask):
    nSlice,nch,nrow,ncol=org.shape
    orgk= np.empty(org.shape,dtype=np.complex64)
    atb = np.empty(org.shape,dtype=np.complex64)
    minv= np.zeros((nSlice,),dtype=np.complex64)
    for i in range(nSlice):
        A  = lambda z: usp(z,mask[i],nch,nrow,ncol)
        At = lambda z: usph(z,mask[i],nch,nrow,ncol)
        orgk[i],y=A(org[i])
        atb[i],minv[i]=At(y)
        orgk[i]=orgk[i]/minv[i]
    del org
    return orgk, atb, minv


def usp3d(x,mask,nch,nt,nrow,ncol):
    """ This is a the A operator as defined in the paper"""
    kspace=np.reshape(x,(nch,nt,nrow,ncol))
    res=kspace[mask!=0]
    return kspace,res

def usph3d(kspaceUnder,mask,nch,nt,nrow,ncol):
    """ This is a the A^T operator as defined in the paper"""
    temp=np.zeros((nch,nt,nrow,ncol),dtype=np.complex64)
    temp[mask!=0]=kspaceUnder
    minv=np.std(temp)
    temp=temp/minv
    return temp,minv

def generateUndersampled3d(org,mask):
    nSlice,nch,nt,nrow,ncol=org.shape
    orgk=np.empty(org.shape,dtype=np.complex64)
    atb=np.empty(org.shape,dtype=np.complex64)
    minv=np.zeros((nSlice,),dtype=np.complex64)
    for i in range(nSlice):
        A  = lambda z: usp3d(z,mask[i],nch,nt,nrow,ncol)
        At = lambda z: usph3d(z,mask[i],nch,nt,nrow,ncol)
        orgk[i],y=A(org[i])
        atb[i],minv[i]=At(y)
        orgk[i]=orgk[i]/minv[i]
    del org
    return orgk,atb,minv

def norm3d(org):
    nSlice,nch,nt,nrow,ncol=org.shape
    minv=np.zeros((nSlice,),dtype=np.complex64)
    for i in range(nSlice):
        minv[i] =np.std(org[i])
        org[i]=org[i]/minv[i]
    return org,minv

def usph2(kspaceUnder,mask,nch,nrow,ncol):
    """ This is a the A^T operator as defined in the paper"""
    temp=np.zeros((nch,nrow,ncol),dtype=np.complex64)
    temp[mask!=0]=kspaceUnder
    return temp

def Ax(org,mask):
    nSlice,nch,nrow,ncol=org.shape
    atb=np.empty(org.shape,dtype=np.complex64)
    for i in range(nSlice):
        A  = lambda z: usp(z,mask[i],nch,nrow,ncol)
        At = lambda z: usph2(z,mask[i],nch,nrow,ncol)
        _,y=A(org[i])
        atb[i]=At(y)
    del org
    return atb

def c2r_3d(inp):
    """  input img: row x col in complex64
    output image: row  x col x2 in float32
    """
    if inp.dtype=='complex64':
        dtype=np.float32
    else:
        dtype=np.float64
    nImg,nCh,nt,nrow,ncol=inp.shape
    out=np.zeros((nImg,nCh*2,nt,nrow,ncol),dtype=dtype)
    out[:,0:nCh,:,:,:]=np.real(inp)
    out[:,nCh:nCh*2,:,:,:]=np.imag(inp)
    return out


def readcfl(name):
    # get dims from .hdr
    h = open(name + ".hdr", "r")
    h.readline() # skip
    l = h.readline()
    h.close()
    dims = [int(i) for i in l.split( )]

    # remove singleton dimensions from the end
    n = np.prod(dims)
    dims_prod = np.cumprod(dims)
    dims = dims[:np.searchsorted(dims_prod, n)+1]

    # load data and reshape into dims
    d = open(name + ".cfl", "r")
    a = np.fromfile(d, dtype=np.complex64, count=n)
    d.close()
    return a.reshape(dims, order='F') # column-major

def IFFTc(x, axis, norm='ortho'):
    ''' expect x as m*n matrix '''
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x, axes=axis), axis=axis, norm=norm), axes=axis)


