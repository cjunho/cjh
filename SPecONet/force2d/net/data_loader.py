#data_loader.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pprint import pprint
import subprocess


def load_obj(name):
    print("poiuy")
    with open('./data/' + name + '.pkl', 'rb') as f:
        data = pickle.load(f)
        data = data[:,:,0,:]
    return data

def get_data1(gparams, kind='train', transform_f=None):
    equation, file, sd = 'Standard', gparams['file'], gparams['sd']
    # input("789")
    # print(file)
    if sd == 1:
        sd = 1.0
    
    shape, epsilon = int(file.split('N')[1]) + 1, gparams['epsilon']
    forcing = 'normal'
    # input("456")
    # print(file)
    if kind == 'validate':
        size = 1000
        file = f'{size}N{shape-1}'
    else:
        size = int(file.split('N')[0])
    # input("098")
    # print(file)
    data = LGDataset(equation=equation, pickle_file=file, shape=shape, kind=kind, sd=sd, forcing=forcing, transform_f=transform_f)
    return data


def get_data(gparams, kind='train', transform_f=None):
    equation, file, sd,dt,ndt = gparams['equation'], gparams['file'], gparams['sd'],gparams['dt'],gparams['ndt']
    # equation, sd,dt,ndt = gparams['equation'], gparams['sd'],gparams['dt'],gparams['ndt']
    # file = f'100N15'
    # input("123")
    # print(equation)
    if sd == 1:
        sd = 1.0
    
    shape, epsilon = int(file.split('N')[1]) + 1, gparams['epsilon']
    # ndata=int(gparams['file'].split('N')[0])
    forcing = gparams['forcing']
    
    # PATH_prev=gparams['PATH_prev']
    # input("qwe456")
    print(equation)
   
    if equation == 'test3d':
        size = 100
        file = f'{size}N{shape-1}'
        
    else:
        size = int(file.split('N')[0])
    # input("789")
    # print(file)
    # try:
    data = LGDataset(equation=equation,dt=dt,ndt=ndt,epsilon=epsilon, pickle_file=file, shape=shape, kind=kind, sd=sd, forcing=forcing, transform_f=transform_f)
        
    # except:        
        
    #     subprocess.call(f'python create_train_data.py --equation {equation} --size {size}'\
    #                     f' --file {file} --N {shape - 1} --eps {epsilon} --kind {kind} --sd {sd} --forcing {forcing} --dt {dt} --ndt {ndt}', shell=True)
        
    #     data = LGDataset(equation=equation,dt=dt,ndt=ndt,epsilon=epsilon, pickle_file=file, shape=shape, kind=kind, sd=sd, forcing=forcing, transform_f=transform_f)
        
    return data

def get_data1(path,shape):
    
    data = LGDataset0(path,shape)
    return data
class LGDataset0():
    def __init__(self,  path,shape):
       
        f=torch.load(path+'/data.pt')
           
            # self.data = pickle.load(f)
        self.data = f
        self.shape = shape
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        u = self.data[idx].double().detach().reshape(3,1,self.shape-2, self.shape-2, self.shape-2)
        
        sample = {'data_u': u}
        return sample
class LGDataset():
    """Legendre-Galerkin Dataset."""    
    def __init__(self, equation,dt,ndt,epsilon, pickle_file, shape=64, transform_f=None, transform_a=None, kind='train', sd=1, forcing='uniform',path=None):
        # print(equation)
        """
        Args:
            pickle_file (string): Path to the pkl file with annotations.
            root_dir (string): Directory with all the images.
        """
        
        # input("qwe456")
        # if forcing == 'uniform':
        #     pickle_file += f'uniform'
        # elif forcing == 'normal':
        #     pickle_file += f'sd{sd}'
        # else: pickle_file += f'zero'
        
        with open(f'./data/{equation}{epsilon}/{kind}/' + pickle_file + f'{forcing}.pkl', 'rb') as f:
           
            self.data = pickle.load(f)
            self.data = self.data[:,:]
            # input("qwe789")
        self.ndt = ndt
        # self.ndata = ndata
        self.epsilon = epsilon
        self.equation = equation
        self.transform_f = transform_f
        self.transform_a = transform_a
        self.shape = shape
        # self.alphi1=torch.load(path+'/alphi.pt').detach().double()
        # DATASET=self.alphi1.shape[0]
        # self.alp1=torch.load(path+'/alpha.pt').reshape(DATASET,3,self.shape-2,self.shape-2,self.shape-2).detach().double()
        
      
    # input("first")
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        # input("second")
        # input("qwe2")
        # print(self.data[:,4][idx].shape)
        #L = self.data[:,1][idx].shape[0] 
        
        L = int(self.data[:,1][idx].shape[0]/3)
        # L = 3*10
        # print(self.data[:,0][idx].shape)
        LL=self.data[:,0][idx].shape[1] 
        # LL=self.ndt
        # LL=10
        # L=1
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        #if self.equation == 'Standard2D':
        if self.equation in ('Standard2D', 'NS2d'):
            # print(torch.from_numpy(self.data[:,0][idx][:,:,:,:]).shape)
            # input('sdfds')
            u = torch.from_numpy(self.data[:,0][idx][:,:,:,:]).double().reshape(3,LL,self.shape-2, self.shape-2)
            # f = torch.from_numpy(self.data[:,1][idx][:,:,:,:]).double().reshape(2,LL, self.shape, self.shape)
            f = torch.from_numpy(self.data[:,1][idx][:,:,:,:]).double().reshape(2,LL, self.shape, self.shape)
            
            uex = torch.from_numpy(self.data[:,2][idx]).double()
            cf=torch.from_numpy(self.data[:,3][idx][:,:,:,:]).double().reshape(2,LL, self.shape-2, self.shape-2)
            cf0=torch.from_numpy(self.data[:,4][idx]).double().reshape(2, self.shape-2, self.shape-2)
            
            sample = {'data_u': u, 'f': f ,'cf':cf,'uex':uex,'cf0':cf0}
            # sample = {'data_u': u, 'f': f ,'uex':uex,'cf':cf}
        elif self.equation in ('test3d'):
            # print(type(self.data[1][idx]))
            # print(self.data[1][idx].shape)
            # print(self.data[:,1][idx].shape)
            # print(idx)
            # print(self.data[:,2][idx].shape)
            # input('okok')
            uex = torch.from_numpy(self.data[:,1][idx]).double()
            f = torch.from_numpy(self.data[:,0][idx]).double().reshape(3,LL, self.shape, self.shape, self.shape)
            u = torch.from_numpy(self.data[:,2][idx][:,:,:,:, :]).double().reshape(4,LL,self.shape-2, self.shape-2, self.shape-2)
            
            
            sample = { 'f': f,'uex':uex,'data_u': u}
        elif self.equation == 'Standardb':            
            u = torch.transpose(torch.Tensor(self.data[:,0][idx]).double().reshape(self.shape,LL),0,1)
            f = torch.Tensor(self.data[:,1][idx]).double().reshape(1, self.shape)
            a = torch.Tensor(self.data[:,2][idx]).double().reshape(1, self.shape-1)
            p = torch.Tensor(self.data[:,3][idx]).double().reshape(1, L)
            Mass = torch.Tensor(self.data[:,5][idx]).double().reshape(self.shape-1, self.shape-1)
            # print(self.data[:,5][idx].shape)
            ff=f
            sample = {'data_u': u, 'f': f, 'a': a, 'p': p, 'fn': ff,'Mass':Mass}
        else:
            
            u = torch.transpose(torch.from_numpy(self.data[:,0][idx]).float().reshape(self.shape,LL),0,1)
            f = torch.from_numpy(self.data[:,1][idx]).double().reshape(1, self.shape)
            a = torch.from_numpy(self.data[:,2][idx]).double().reshape(1, self.shape-2)
            p = torch.from_numpy(self.data[:,3][idx]).double().reshape(1, L)
            # Mass = torch.Tensor([self.data[:,5][idx]]).reshape(self.shape-1, self.shape-1)
            
            ff=f
            sample = {'data_u': u, 'f': f, 'a': a, 'p': p, 'fn': ff}
            # else:
            # sample = {'u': u,'uu': uu, 'f': f, 'a': a, 'p': p}
        return sample


def normalize(gparams, loader):
    from torchvision import transforms
    channels_sum, channels_squares_sum, num_batches = 0, 0, 0

    for _, data in enumerate(loader):
        f = data['f']
        channels_sum += torch.mean(f, dim=[0, 2])
        channels_squares_sum += torch.mean(f**2, dim=[0,2])
        num_batches += 1

    mean = channels_sum/num_batches
    std = (channels_squares_sum/num_batches - mean**2)**0.5    
    gparams['mean'] = float(mean[0].item())
    gparams['std'] = float(std[0].item())
    return gparams, transforms.Normalize(mean, std)

# device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
# class UnitGaussianNormalizer(object):
#     def __init__(self, x, eps=0.00001):
#         super(UnitGaussianNormalizer, self).__init__()

#         # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
#         self.mean = torch.mean(x, 0)
#         self.std = torch.std(x, 0)
#         self.eps = eps
#         # print(x.shape)
#         # input('ok')
#     def encode(self, x):
#         print(x.shape)
#         print(self.mean.shape)
#         print(self.std.shape)
#         print(self.eps)
#         input('inside')
#         x = (x - self.mean) / (self.std + self.eps)
#         return x

#     def decode(self, x, sample_idx=None):
#         if sample_idx is None:
#             std = self.std + self.eps # n
#             mean = self.mean
#         else:
#             if len(self.mean.shape) == len(sample_idx[0].shape):
#                 std = self.std[sample_idx] + self.eps  # batch*n
#                 mean = self.mean[sample_idx]
#             if len(self.mean.shape) > len(sample_idx[0].shape):
#                 std = self.std[:,sample_idx]+ self.eps # T*batch*n
#                 mean = self.mean[:,sample_idx]

#         # x is in shape of batch*n or T*batch*n
#         std1=std.to(device)
#         mean1=mean.to(device)
       
#         x = (x * std1) + mean1
#         return x

#     def cuda(self):
#         self.mean = self.mean.cuda()
#         self.std = self.std.cuda()

#     def cpu(self):
#         self.mean = self.mean.cpu()
#         self.std = self.std.cpu()
