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




def get_data(gparams, kind='train', transform_f=None):
    file,dt,ndt =  gparams['file'], gparams['dt'],gparams['ndt']
   
    shape, epsilon = int(file.split('N')[1]) + 1, gparams['epsilon']
    # ndata=int(gparams['file'].split('N')[0])
    forcing = gparams['forcing']
    
    
    size = int(file.split('N')[0])
    # input("789")
    # print(file)
    # try:
    data = LGDataset(dt=dt,ndt=ndt,epsilon=epsilon, pickle_file=file, shape=shape, kind=kind, forcing=forcing, transform_f=transform_f)
        
     
    return data



class LGDataset():
    """Legendre-Galerkin Dataset."""    
    def __init__(self,dt,ndt,epsilon, pickle_file, shape=64, transform_f=None, transform_a=None, kind='train',  forcing='uniform',path=None):
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
        
        with open(f'./data/NS3d{epsilon}/{forcing}/' + pickle_file + f'{forcing}.pkl', 'rb') as f:
           
            self.data = pickle.load(f)
            self.data = self.data[:,:]
            # input("qwe789")
        self.ndt = ndt
        # self.ndata = ndata
        self.epsilon = epsilon
        

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
        LL=self.data[:,0][idx].shape[1] 
        if torch.is_tensor(idx):
            idx = idx.tolist()

        u = torch.from_numpy(self.data[:,0][idx][:,:,:,:, :]).double().reshape(4,LL,self.shape-2, self.shape-2, self.shape-2)
           
        f = torch.from_numpy(self.data[:,1][idx][:,:,:,:, :]).double().reshape(3,LL, self.shape, self.shape, self.shape)
        cf0=torch.from_numpy(self.data[:,2][idx]).double().reshape(3, self.shape-2, self.shape-2, self.shape-2)
        cf=torch.from_numpy(self.data[:,3][idx][:,:,:,:, :]).double().reshape(3,LL, self.shape-2, self.shape-2, self.shape-2)
        uex = torch.from_numpy(self.data[:,4][idx]).double()
        
        sample = {'data_u': u, 'f': f,'cf0':cf0 ,'cf':cf,'uex':uex}
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

