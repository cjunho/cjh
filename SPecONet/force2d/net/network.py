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
def swish(x,beta):
    return x * torch.sigmoid(beta*x)


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


class ResNet(nn.Module):
    def __init__(self, d_in, filters, d_out, kernel_size=5, padding=2, blocks=0):
        super(ResNet, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.blocks = blocks
        self.filters = filters
        self.conv = conv1d(d_in, self.filters, kernel_size=kernel_size, padding=padding)
        # self.n1 = nn.GroupNorm(1, self.filters)
        # self.relu = nn.ReLU(inplace=True)
        # self.conv1 = conv1d(self.filters, self.filters, kernel_size=kernel_size, padding=padding)
        # self.n2 = nn.GroupNorm(1, self.filters)
        # self.conv2 = conv1d(self.filters, self.filters, kernel_size=kernel_size, padding=padding)
        # self.residual = nn.Sequential(
        #     self.n1,
        #     self.relu,
        #     self.conv1,
        #     self.n2,
        #     self.relu,
        #     self.conv2)
        self.fc1 = nn.Linear(self.filters*(self.d_out + 2), self.d_out, bias=True)
    def forward(self, x):
        out = self.conv(x)
        out = out.flatten(start_dim=1)
        out = self.fc1(out)
        out = out.view(out.shape[0], 1, self.d_out)
        return out


class ResNetD(nn.Module):
    def __init__(self, d_in, filters, d_out, kernel_size=5, padding=2, blocks=5):
        super(ResNetD, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.blocks = blocks
        self.filters = filters
        self.conv = conv1d(d_in, self.filters, kernel_size=kernel_size, padding=1)
        self.n1 = nn.GroupNorm(1, self.filters)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv1d(self.filters, self.filters, kernel_size=kernel_size, padding=0)
        self.n2 = nn.GroupNorm(1, self.filters)
        self.conv2 = conv1d(self.filters, self.filters, kernel_size=kernel_size, padding=0)
        self.residual = nn.Sequential(
            self.n1,
            self.relu,
            self.conv1,
            self.n2,
            self.relu,
            self.conv2)
        # self.fc1 = nn.Linear(self.filters*(self.d_out + 2), self.d_out, bias=True)
        self.fc1 = nn.Linear(self.filters*(self.d_out + 2), self.d_out, bias=True)
    def forward(self, x):
        out = self.conv(x) #1
        if self.blocks != 0:
            for block in range(self.blocks):
                out = self.relu(out + self.residual(out))
        # out = self.n1(out)
        out = out.flatten(start_dim=1)
        out = self.fc1(out)
        out = out.view(out.shape[0], 1, self.d_out)
        return out


class NetA(nn.Module) :
    def __init__(self, d_in, filters, d_out, kernel_size=7, padding=3, blocks=0) :
        super(NetA,self).__init__()
        self.d_in = d_in
        self.blocks = blocks
        self.filters = filters
        self.d_out = d_out
        self.kern = kernel_size
        self.pad = padding
        self.conv1 = conv1d(d_in, filters, kernel_size=self.kern, padding=self.pad)
        self.convH = conv1d(filters, filters, kernel_size=self.kern, padding=self.pad)
        self.fcH = nn.Linear(filters*(self.d_out + 2), self.d_out, bias=True)
    def forward(self, x):
        # print(type(self.conv1))
        # input("fhshshdfg")
        out = F.relu(self.conv1(x))
        if self.blocks != 0:
            for block in range(self.blocks):
                out = F.relu(self.convH(out))
        out = self.convH(out)        
        out = out.flatten(start_dim=1)        
        out = self.fcH(out)       
        # torch.nn.Dropout(0.2)
        out = out.view(out.shape[0], 1, self.d_out)        
        return out
    
class NetA1(nn.Module) :
    def __init__(self, d_in, filters, d_out, kernel_size=7, padding=3, blocks=0) :
        super(NetA1,self).__init__()
        self.d_in = d_in
        self.blocks = blocks
        self.filters = filters
        self.d_out = d_out
        self.kern = kernel_size
        self.pad = padding
        self.conv1 = conv1d(d_in, filters, kernel_size=self.kern, padding=self.pad)
        self.convH = conv1d(filters, filters, kernel_size=self.kern, padding=self.pad)
        self.fcH = nn.Linear(filters*(self.d_out + 1), self.d_out, bias=True)
    # print(filters*(d_out + 1))
    # print(d_out)
    def forward(self, x):
        # print(type(self.conv1))
        # input("fhshshdfg")
        out = F.relu(self.conv1(x))
        if self.blocks != 0:
            for block in range(self.blocks):
                out = F.relu(self.convH(out))
        out = self.convH(out)        
        out = out.flatten(start_dim=1)        
        out = self.fcH(out)        
        # torch.nn.Dropout(0.2)
        out = out.view(out.shape[0], 1, self.d_out)        
        return out


class NetB(nn.Module) :
    def __init__(self, d_in, filters, d_out, kernel_size=7, padding=3, blocks=0) :
        super(NetB,self).__init__()
        self.d_in = d_in
        self.blocks = blocks
        self.filters = filters
        self.d_out = d_out
        self.kern = kernel_size
        self.pad = padding
        self.conv1 = conv1d(d_in, filters, kernel_size=self.kern, padding=self.pad)
        self.convH = conv1d(filters, filters, kernel_size=self.kern, padding=self.pad)
        self.fcH = nn.Linear(filters*(self.d_out + 2), self.d_out, bias=True)
    def forward(self, x):
        m = nn.Sigmoid()
        out = m(self.conv1(x))
        if self.blocks != 0:
            for block in range(self.blocks):
                out = m(self.convH(out))
        out = self.convH(out)
        out = out.flatten(start_dim=1)
        out = self.fcH(out)
        out = out.view(out.shape[0], 1, self.d_out)
        return out


class NetC(nn.Module) :
    def __init__(self, d_in, filters, d_out, kernel_size=7, padding=3, blocks=0) :
        super(NetC,self).__init__()
        self.d_in = d_in
        self.blocks = blocks
        self.filters = filters
        self.d_out = d_out
        self.swish = swish
        self.kern = kernel_size
        self.pad = padding
        self.conv1 = conv1d(d_in, filters, kernel_size=self.kern, padding=self.pad)
        self.convH = conv1d(filters, filters, kernel_size=self.kern, padding=self.pad)
        self.fcH = nn.Linear(filters*(self.d_out + 2), self.d_out, bias=True)
    def forward(self, x):
        m = self.swish
        out = m(self.conv1(x))
        if self.blocks != 0:
            for block in range(self.blocks):
                out = m(self.convH(out))
        out = self.convH(out)
        out = out.flatten(start_dim=1)
        out = self.fcH(out)
        out = out.view(out.shape[0], 1, self.d_out)
        return out


# class NetD(nn.Module) :
#     def __init__(self, d_in, filters, d_out, kernel_size=5, padding=0, blocks=0, activation='swish'):
#         super(NetD, self).__init__()
#         self.d_in = d_in
#         self.blocks = blocks
#         self.filters = filters
#         self.activation = activation.lower()
#         self.m = swish
#         self.d_out = d_out
#         self.swish = swish
#         self.kern = kernel_size
#         self.pad = padding
#         self.conv1 = conv1d(d_in, filters, kernel_size=self.kern, padding=1)
#         # self.pool = nn.AdaptiveMaxPool1d(1)
#         self.convH = conv1d(filters, filters, kernel_size=self.kern, padding=0)
#         self.dim = d_in*(d_out - 4*(self.blocks + 1))*filters
#         self.fcH = nn.Linear(24, self.d_out+2, bias=False)
#         # self.fcH = nn.Linear(32, self.d_out, bias=True)        
#         print(self.dim)
#         print(self.d_out)
#     def forward(self, x):
#         # if self.activation == 'relu':
#         #     m = self.relu
#         # elif self.activation == 'sigmoid':
#         #     m = self.sigmoid
#         # elif self.activation == 'swish':
#         m = self.swish
#         print('111',x.shape)
#         out = m(self.conv1(x))
#         print('222',out.shape)
#         if self.blocks != 0:
#             for block in range(self.blocks):
#                 out = m(self.convH(out))
#         print('333',out.shape)
#         out = self.convH(out)
#         # out = self.pool(out)
#         print('444',out.shape)
#         out = out.flatten(start_dim=1)
#         # print(out.shape)
#         print('555',out.shape)
#         out = self.fcH(out)
#         print('666',out.shape)
#         out = out.view(out.shape[0], 1, self.d_out+2)
#         return out
class NetD(nn.Module) :
    def __init__(self, d_in, filters, d_out, kernel_size=5, padding=0, blocks=0, activation='swish'):
        super(NetD, self).__init__()
        self.d_in = d_in
        self.blocks = blocks
        self.filters = filters
        self.activation = activation.lower()
        self.m = swish
        self.d_out = d_out
        self.swish = swish
        self.kern = kernel_size
        self.pad = padding
        # self.conv1 = conv1d(d_in, filters, kernel_size=self.kern, padding=1)
        # # self.pool = nn.AdaptiveMaxPool1d(1)
        # self.convH = conv1d(filters, filters, kernel_size=self.kern, padding=0)
        self.dim = d_in*(d_out - 4*(self.blocks + 1))*filters
        self.fcH = nn.Linear(self.d_out+2,self.d_out+2, bias=False)
        # self.fcH1 = nn.Linear(self.d_out+2,self.d_out+2, bias=False)
        # self.fcH2 = nn.Linear(self.d_out+2,self.d_out+2, bias=False)
        # self.fcH = nn.Linear(32, self.d_out, bias=True)        
        # print(self.dim)
        # print(self.d_out)
    def forward(self, x):
        # if self.activation == 'relu':
        #     m = self.relu
        # elif self.activation == 'sigmoid':
        #     m = self.sigmoid
        # elif self.activation == 'swish':
        # m = self.swish
        # print('111',x.shape)
        # out = m(self.conv1(x))
        # out = self.conv1(x)
        # print('222',out.shape)
        # if self.blocks != 0:
        #     for block in range(self.blocks):
        #         out = m(self.convH(out))
        # print('333',out.shape)
        # out = self.convH(out)
        # # out = self.pool(out)
        # print('444',out.shape)
        out=x
        # out = out.flatten(start_dim=1)
        # print(out.shape)
        # print('555',out.shape)
        out = self.fcH(out)
        # out=torch.transpose(torch.transpose(self.fcH1(out),2,0),2,1)
        # out=torch.transpose(self.fcH2(out),2,1)
        # print('666',out.shape)
        # out = out.view(out.shape[0],1, self.d_out+2)
        return out


class FC(nn.Module):
    def __init__(self, d_in, hidden, d_out, layers=1, activation='relu') :
        super(FC, self).__init__()
        self.layer1 = nn.Linear(1, 32)
        self.relu = nn.ReLU(inplace=True)
        self.swish = swish
        self.sigmoid = nn.Sigmoid
        self.hidden = hidden
        self.layers = layers
        self.activation = activation
        self.d_in = d_in
        self.d_out = d_out
        self.layer2 = nn.Linear(hidden, hidden)
        self.layer3 = nn.Linear(hidden, d_out)
    def forward(self, x):
        if self.activation.lower() == 'relu':
            m = self.relu
        elif self.activation.lower() == 'sigmoid':
            m = self.sigmoid
        elif self.activation.lower() == 'swish':
            m = self.swish
        out = self.relu(self.layer2(x))
        for _ in range(self.layers):
            out = self.relu(self.layer2(out))
        out = self.layer3(out)
        out = out.view(out.shape[0], self.d_in, self.d_out)
        return out



class Net2D(nn.Module) :
    def __init__(self, d_in, filters, d_out, kernel_size=7, padding=3, blocks=0) :
        super(Net2D, self).__init__()
        self.d_in = d_in
        self.blocks = blocks
        self.filters = filters
        self.d_out = d_out
        self.swish = swish
        # self.swish = nn. ReLU()
        self.kern = kernel_size
        self.pad = padding
        self.conv1 = conv2d(d_in, filters, kernel_size=self.kern, padding=self.pad)
        self.convH = conv2d(filters, filters, kernel_size=self.kern, padding=self.pad)
        # print(self.filters*(self.d_out + 2))
        self.fcH = nn.Linear(filters*(self.d_out)**2, self.d_out**3, bias=True)
        # self.fcH = nn.Linear(32768, self.d_out**2, bias=True)

    def forward(self, x):
        m = self.swish
        out = m(self.conv1(x))
        if self.blocks != 0:
            for block in range(self.blocks):
                out = m(self.convH(out))
        out = self.convH(out)
        out = out.flatten(start_dim=1)
        # print(out.shape)
        out = self.fcH(out)
        out = out.view(out.shape[0], 1,self.d_out,self.d_out,self.d_out)
        return out



class Net3D(nn.Module) :
    def __init__(self,beta,ndt, d_in, filters, d_out, kernel_size=7, padding=3, blocks=0) :
        super(Net3D, self).__init__()
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
        
        # self.conv2 = conv3d(self.filters, self.filters, kernel_size=7, padding=self.pad)
        
        # self.conv3 = conv3d(self.filters, self.filters, kernel_size=7, padding=self.pad)
        
        # self.conv4 = conv3d(self.filters, self.filters, kernel_size=7, padding=self.pad)
        #self.convH = conv3d(filters, filters, kernel_size=7, padding=self.pad)
        # print(self.filters*(self.d_out + 2))
        # self.fcH = nn.Linear(filters*self.Dout**3 ,int((self.d_out)**3*4*self.ndt), bias=False)
        self.fcH = nn.Linear(int(self.filters*self.d_out**2) ,int(2*(self.d_out)**2*self.ndt), bias=False)
        # self.fcH = nn.Linear(32768, self.d_out**2, bias=True)
        
    def forward(self, x):
        
        m = self.swish
       
        out = m(self.conv1(x),self.beta)
        
        # out = m(self.conv2(out))
        
        # out = m(self.conv3(out))
        
        # out = m(self.conv4(out))
        
        #if self.blocks != 0:
          #   for block in range(self.blocks):
          #      out = m(self.convH(out))
        
        # print('33',out.shape)
        # out = out.flatten(start_dim=2)
        
        out=out.view(out.shape[0],self.filters*self.d_out**2)
            #+10**-4*torch.eye(out.shape[0],self.filters*self.d_out**3,requires_grad=False, device="cuda",dtype=torch.float64)

       
        out = self.fcH(out)
        # # # # print('55',out.shape)
        
        out = out.view(out.shape[0],2, self.ndt,self.d_out,self.d_out)
        # print('66',out.shape)
        # input('fdgs')
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
        
        # self.conv2 = conv3d(self.filters, self.filters, kernel_size=7, padding=self.pad)
        
        # self.conv3 = conv3d(self.filters, self.filters, kernel_size=7, padding=self.pad)
        
        # self.conv4 = conv3d(self.filters, self.filters, kernel_size=7, padding=self.pad)
        #self.convH = conv3d(filters, filters, kernel_size=7, padding=self.pad)
        # print(self.filters*(self.d_out + 2))
        # self.fcH = nn.Linear(filters*self.Dout**3 ,int((self.d_out)**3*4*self.ndt), bias=False)
        self.fcH = nn.Linear(int(self.filters*self.d_out**2) ,int(2*(self.d_out)**2*self.ndt), bias=False)
        # self.fcH = nn.Linear(32768, self.d_out**2, bias=True)
        
    def forward(self, x):
        
        m = self.swish
       
        out = m(self.conv1(x),self.beta)
        
        # out = m(self.conv2(out))
        
        # out = m(self.conv3(out))
        
        # out = m(self.conv4(out))
        
        #if self.blocks != 0:
          #   for block in range(self.blocks):
          #      out = m(self.convH(out))
        
        # print('33',out.shape)
        # out = out.flatten(start_dim=2)
      
        out=out.view(out.shape[0],self.filters*self.d_out**2)
            #+10**-4*torch.eye(out.shape[0],self.filters*self.d_out**3,requires_grad=False, device="cuda",dtype=torch.float64)

        return out

class Net3T(nn.Module) :
    def __init__(self,ndt, d_in, filters, d_out, kernel_size=7, padding=3, blocks=0) :
        super(Net3T, self).__init__()
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
        self.ll=   int(self.filters/3) 
    
        self.conv1 = conv3d(self.d_in, self.filters, kernel_size=self.kern, padding=self.pad)
        #self.convH = conv3d(filters, filters, kernel_size=7, padding=self.pad)
        # print(self.filters*(self.d_out + 2))
        # self.fcH = nn.Linear(filters*self.Dout**3 ,int((self.d_out)**3*4*self.ndt), bias=False)
        self.fcH1 = nn.Linear(int(self.ll/self.ndt*self.d_out**3) ,int((self.d_out)**3), bias=False)
        self.fcH2 = nn.Linear(int(self.ll/self.ndt*self.d_out**3) ,int((self.d_out)**3), bias=False)
        # self.fcH3 = nn.Linear(int(self.ll*self.d_out**3) ,int((self.d_out)**3*self.ndt), bias=False)
        # self.fcH4 = nn.Linear(int(self.ll*self.d_out**3) ,int((self.d_out)**3*self.ndt), bias=False)
        # self.fcH5 = nn.Linear(int(self.ll*self.d_out**3) ,int((self.d_out)**3*self.ndt), bias=False)
        # self.fcH = nn.Linear(32768, self.d_out**2, bias=True)
        
    def forward(self, x):
        
        m = self.swish
        
        out = m(self.conv1(x))
        
        #if self.blocks != 0:
          #   for block in range(self.blocks):
          #      out = m(self.convH(out))
        
        # print('33',out.shape)
        # out = out.flatten(start_dim=2)
        # out=out.view(out.shape[0],self.ll,self.d_out**3)
        # out1=out[:,:21]
        out=out.view(3*out.shape[0],14*self.d_out**3)
        # out2=out[:,21:].reshape(3*out.shape[0],7*self.d_out**3)
        # print('44',out.shape)
        # out1=out[:,:7*self.d_out**3]
        
        out1 = self.fcH1(out[:,:7*self.d_out**3])
        # print('55',out1.shape)
        
        out1 = out1.view(out1.shape[0], 1,self.d_out,self.d_out,self.d_out)
        
        # out2 = self.fcH2(out2)
        # print('55',out2.shape)
        out2=0
        # out2 = out2.view(out2.shape[0], 1,self.d_out,self.d_out,self.d_out)
        # print('66',out.shape)
        # input('fdgs')
        return out1,out2

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
        # print('22',out.shape)
        #if self.blocks != 0:
         #   for block in range(self.blocks):
          #      out = m(self.convH(out))
        
        # print('33',out.shape)
        # out = out.flatten(start_dim=2)
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
        # print('22',out.shape)
        #if self.blocks != 0:
         #   for block in range(self.blocks):
          #      out = m(self.convH(out))
        # print(self.Dout)
        # print(self.d_out)
        # print(self.filters)
        # print('33',out.shape)
        # 
        # out1 = out.flatten(start_dim=1)
       
        out=out.view(out.shape[0],self.filters*self.Dout**2)
        # print(out1.shape)
        # print(out2.shape)
        # err=abs(out1-out2)
        
        # print('44',torch.max(err))
        out = self.fcH(out)
        # print('55',out.shape)
        # input('fdgs')
        out = out.view(out.shape[0], self.ndt,self.d_out,self.d_out)
        # print('66',out.shape)
        # input('fdgs')
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
        # print('22',out.shape)
        #if self.blocks != 0:
         #   for block in range(self.blocks):
          #      out = m(self.convH(out))
        # print(self.Dout)
        # print(self.d_out)
        # print(self.filters)
        # print('33',out.shape)
        # 
        # out1 = out.flatten(start_dim=1)
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
        # print('22',out.shape)
        #if self.blocks != 0:
         #   for block in range(self.blocks):
          #      out = m(self.convH(out))
        # print(self.Dout)
        # print(self.d_out)
        # print(self.filters)
        # print('33',out.shape)
        # 
        # out1 = out.flatten(start_dim=1)
        out=out.view(out.shape[0],self.filters*self.Dout**3)
        # print(out1.shape)
        # print(out2.shape)
        # err=abs(out1-out2)
        
        # print('44',torch.max(err))
        # out = self.fcH(out)
        # print('55',out.shape)
        # input('fdgs')
        # out = out.view(out.shape[0], self.ndt,self.d_out,self.d_out,self.d_out)
        # print('66',out.shape)
        # input('fdgs')
        return out



class Net3Dlin(nn.Module) :
    def __init__(self,ndt, d_in, filters, d_out, kernel_size=7, padding=3, blocks=0) :
        super(Net3Dlin, self).__init__()
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
        self.conv1 = conv3d(d_in, filters, kernel_size=self.kern, padding=self.pad)
        self.convH = conv3d(filters, filters, kernel_size=7, padding=self.pad)
        # print(self.filters*(self.d_out + 2))
        # self.fcH = nn.Linear(filters*self.Dout**3 ,int((self.d_out)**3*self.ndt), bias=False)
        self.fcH = nn.Linear(self.d_out**3 ,int((self.d_out)**3*self.ndt), bias=False)
        # self.fcH = nn.Linear(32768, self.d_out**2, bias=True)
        
    def forward(self, x):
        
        m = self.swish
        # print('11',x.shape)
        out1 = m(self.conv1(x))
        # print('22',out.shape)
        if self.blocks != 0:
            for block in range(self.blocks):
                out1 = m(self.convH(out1))
        
        # print('33',out.shape)
        # out = out.flatten(start_dim=2)
        out=out1.view(4*out1.shape[0],self.d_out**3)
        # print('44',out.shape)
        # out = self.fcH(out)
        # print('55',out.shape)
        
        # out = out.view(out.shape[0], self.ndt,self.d_out,self.d_out,self.d_out)
        # print('66',out.shape)
        # input('fdgs')
        return out,out1

# class Net2D(nn.Module) : # Linear
#     def __init__(self, d_in, filters, d_out, kernel_size=7, padding=3, blocks=0) :
#         super(Net2D, self).__init__()
#         self.d_in = d_in
#         self.blocks = blocks
#         self.filters = filters
#         self.d_out = d_out
#         # self.swish = swish
#         self.swish = nn.ReLU()
#         self.kern = kernel_size
#         self.pad = padding
#         self.conv1 = conv2d(d_in, filters, kernel_size=self.kern, padding=self.pad)
#         self.fcH = nn.Linear(filters*(self.d_out+2)**2, self.d_out**2, bias=True)

#     def forward(self, x):
#         out = self.conv1(x)
#         out = out.flatten(start_dim=1)
#         out = self.fcH(out)
#         out = out.view(out.shape[0], 1, self.d_out, self.d_out)
#         return out

