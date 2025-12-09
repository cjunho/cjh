import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# KAIMING INITIALIZATION
def weights_init(m):
    if isinstance(m, nn.Conv1d):
        # torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.kaiming_normal_(m.weight.data)
        torch.nn.init.zeros_(m.bias)

def weights_xavier(m):
    if isinstance(m, nn.Conv1d):
        # torch.nn.init.kaiming_normal_(m.weight.data)
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)    

def init_optim(model):
    params = {'history_size': 5,
              'tolerance_grad': 1E-15,
              'tolerance_change': 1E-15,
              'max_eval': 10,
                }
    # params = { 'lr':0.001
    #             }
    # return torch.optim.SGD(model.parameters(), **params)
    return torch.optim.LBFGS(model.parameters(), **params)
    # return torch.optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-10, weight_decay=0, amsgrad=False)
    # return torch.optim.Adam
def swish(x):
    return x * torch.sigmoid(x)


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-9):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss


class RelMSELoss(nn.Module):
    def __init__(self, batch):
        super().__init__()
        self.mse = nn.MSELoss()
        self.batch = batch
    def forward(self,yhat,y):
        loss = self.mse(yhat,y)/self.batch
        return loss


def conv1d(in_planes, out_planes, stride=1, bias=False, kernel_size=5, padding=2, dialation=1) :
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

def conv2d(in_planes, out_planes, stride=1, bias=True, kernel_size=5, padding=2, dialation=1) :
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

def conv3d(in_planes, out_planes, stride=1, bias=False, kernel_size=5, padding=2, dialation=1) :
    return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

class Linear(nn.Module):
    def __init__(self, d_in, filters, d_out, kernel_size=5, padding=2, blocks=0):
        super(Linear, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.blocks = blocks
        self.filters = filters
        self.conv = conv1d(d_in, self.filters, kernel_size=kernel_size, padding=padding)
        # self.fc1 = nn.Linear(self.filters*(self.d_out + 2), self.d_out, bias=True)
        self.fc1 = nn.Linear((self.d_out + 2), (self.d_out + 2), bias=False)
    def forward(self, x):
        # out = self.conv(x)
        out = x.flatten(start_dim=1)
        # out = self.fc1(out)
        out = self.fc1(out)
        out = out.view(out.shape[0], 1, self.d_out+2)
        return out


class Net3D(nn.Module) :
    def __init__(self,ndt, d_in, filters, d_out, kernel_size=7, padding=3, blocks=0) :
        super(Net3D, self).__init__()
        self.d_in = d_in
        self.blocks = blocks
        self.filters = filters
        self.d_out = d_out
        self.swish = swish
        
        # self.swish = nn. ReLU()
        self.kern = kernel_size
        self.pad = padding
        self.Dout=self.d_out+2+(self.blocks+2)*(2*self.pad-self.kern+1)
        # print(self.d_in)
        # print(self.Dout)
        # print(3*ndt*filters*self.Dout**3)
        self.ndt=ndt
        # self.Dout=12
        # self.ll=   int(self.filters/3) 
    
        self.conv1 = conv2d(self.d_in, self.filters, kernel_size=self.kern, padding=self.pad, bias=True)
        
        self.fcH = nn.Linear(int(self.filters*self.d_out**2) ,int(2*(self.d_out)**2*self.ndt), bias=False)
        
    def forward(self, x):
        
        m = self.swish
       
        out = m(self.conv1(x))
        
       
        
        out=out.view(out.shape[0],self.filters*self.d_out**2)
        
       
        out = self.fcH(out)
       
        
        out = out.view(out.shape[0],2, self.ndt,self.d_out,self.d_out)
       
        return out

class Net3D0(nn.Module) :
    def __init__(self,beta,ndt, d_in, filters, d_out, kernel_size=7, padding=3, blocks=0) :
        super(Net3D0, self).__init__()
        self.d_in = d_in
        self.blocks = blocks
        self.filters = filters
        self.d_out = d_out
        self.swish = swish
        self.beta=beta
        # self.swish = nn. ReLU()
        self.kern = kernel_size
        self.pad = padding
        self.Dout=self.d_out+2+(self.blocks+2)*(2*self.pad-self.kern+1)
        # print(self.d_in)
        # print(self.Dout)
        # print(3*ndt*filters*self.Dout**3)
        self.ndt=ndt
        # self.Dout=12
        # self.ll=   int(self.filters/3) 
    
        self.conv1 = conv2d(self.d_in, self.filters, kernel_size=self.kern, padding=self.pad, bias=True)
        
        self.fcH = nn.Linear(int(self.filters*self.d_out**2) ,int(2*(self.d_out)**2*self.ndt), bias=False)
       
        
    def forward(self, x):
        
        m = self.swish
       
        out = m(self.conv1(x),self.beta)
        
       
      
        out=out.view(out.shape[0],self.filters*self.d_out**2)
        
        return out



class Net3D1(nn.Module) :
    def __init__(self,ndt, d_in, filters, d_out, kernel_size=7, padding=3, blocks=0) :
        super(Net3D1, self).__init__()
        self.d_in = d_in
        self.blocks = blocks
        self.filters = filters
        self.d_out = d_out
        self.swish = swish
        
        # self.swish = nn. ReLU()
        self.kern = kernel_size
        self.pad = padding
        self.Dout=self.d_out+2+(self.blocks+2)*(2*self.pad-self.kern+1)
        # print(self.d_in)
        # print(self.Dout)
        # print(3*ndt*filters*self.Dout**3)
        self.ndt=ndt
        # self.Dout=12
        self.ll=   int(self.filters/4) 
    
        self.conv1 = conv3d(self.d_in, self.filters, kernel_size=self.kern, padding=self.pad)
        #self.convH = conv3d(filters, filters, kernel_size=7, padding=self.pad)
        # print(self.filters*(self.d_out + 2))
        # self.fcH = nn.Linear(filters*self.Dout**3 ,int((self.d_out)**3*4*self.ndt), bias=False)
        self.fcH = nn.Linear(int(self.ll*self.d_out**3) ,int((self.d_out)**3*self.ndt), bias=False)
        # self.fcH = nn.Linear(32768, self.d_out**2, bias=True)
        
    def forward(self, x):
        
        m = self.swish
        # print('11',x.shape)
        out = m(self.conv1(x))
        
        out=out.view(4*out.shape[0],self.ll*self.d_out**3)
        # print('44',out.shape)
        out = self.fcH(out)
        # print('55',out.shape)
        
        out = out.view(out.shape[0], self.ndt,self.d_out,self.d_out,self.d_out)
        # print('66',out.shape)
        # input('fdgs')
        return out
    
class Net3Dpressure(nn.Module) :
    def __init__(self,beta,ndt, d_in, filters, d_out, kernel_size=7, padding=3, blocks=0) :
        super(Net3Dpressure, self).__init__()
        self.d_in = d_in
        self.blocks = blocks
        self.filters = filters
        self.d_out = d_out
        self.swish = swish
        self.beta=beta
        # self.swish = nn. ReLU()
        self.kern = kernel_size
        self.pad = padding
        self.Dout=self.d_out+2+(2*self.pad-self.kern+1)
        # print(self.d_in)
        # print(self.Dout)
        # print(3*ndt*filters*self.Dout**3)
        self.ndt=ndt
        # self.Dout=12
        # self.ll=   int(self.filters) 
    
        self.conv1 = conv2d(self.d_in, self.filters, kernel_size=self.kern, padding=self.pad, bias=True)
        #self.convH = conv3d(filters, filters, kernel_size=7, padding=self.pad)
        # print(self.filters*(self.d_out + 2))
        # self.fcH = nn.Linear(filters*self.Dout**3 ,int((self.d_out)**3*4*self.ndt), bias=False)
        self.fcH = nn.Linear(int(self.filters*self.Dout**2) ,int((self.d_out)**2*self.ndt), bias=False)
        # self.fcH = nn.Linear(32768, self.d_out**2, bias=True)
        
    def forward(self, x):
        
        m = self.swish
        
        # print('11',x.shape)
        out = m(self.conv1(x),self.beta)
        
       
        out=out.view(out.shape[0],self.filters*self.Dout**2)
        
        
        
        out = self.fcH(out)
       
        out = out.view(out.shape[0], self.ndt,self.d_out,self.d_out)
     
        return out


class Net3Dpressure0(nn.Module) :
    def __init__(self,beta,ndt, d_in, filters, d_out, kernel_size=7, padding=3, blocks=0) :
        super(Net3Dpressure0, self).__init__()
        self.d_in = d_in
        self.blocks = blocks
        self.filters = filters
        self.d_out = d_out
        self.swish = swish
        self.beta=beta
        # self.swish = nn. ReLU()
        self.kern = kernel_size
        self.pad = padding
        self.Dout=self.d_out+2+(2*self.pad-self.kern+1)
        
        self.ndt=ndt
        # self.Dout=12
        # self.ll=   int(self.filters) 
    
        self.conv1 = conv2d(self.d_in, self.filters, kernel_size=self.kern, padding=self.pad, bias=True)
        self.fcH = nn.Linear(int(self.filters*self.Dout**2) ,int((self.d_out)**2*self.ndt), bias=False)
       
        
    def forward(self, x):
        
        m = self.swish
        
        # print('11',x.shape)
        out = m(self.conv1(x),self.beta)
        
        out=out.view(out.shape[0],self.filters*self.Dout**2)
        
        return out


class Net3Dpressure1(nn.Module) :
    def __init__(self,beta,ndt, d_in, filters, d_out, kernel_size=7, padding=3, blocks=0) :
        super(Net3Dpressure1, self).__init__()
        self.d_in = d_in
        self.blocks = blocks
        self.filters = filters
        self.d_out = d_out
        self.swish = swish
        self.beta=beta
        # self.swish = nn. ReLU()
        self.kern = kernel_size
        self.pad = padding
        self.Dout=self.d_out+2+(2*self.pad-self.kern+1)
        # print(self.d_in)
        # print(self.Dout)
        # print(3*ndt*filters*self.Dout**3)
        self.ndt=ndt
        # self.Dout=12
        # self.ll=   int(self.filters) 
    
        self.conv1 = conv3d(self.d_in, self.filters, kernel_size=self.kern, padding=self.pad, bias=True)
        
        
    def forward(self, x):
        
        m = self.swish
        
        # print('11',x.shape)
        out = m(self.conv1(x),self.beta)
        
        out=out.view(out.shape[0],self.filters*self.Dout**3)
        
        return out



