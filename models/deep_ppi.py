#-*- encoding:utf8 -*-

import os
import time
import sys

import torch as t
from torch import nn
# from torch.autograd import Variable
# from torchsummary import summary

#from basic_module import BasicModule
from models.BasicModule import BasicModule

sys.path.append("../")
from utils.config import DefaultConfig
configs = DefaultConfig()

       
class ConvsLayer(BasicModule):
    def __init__(self,):

        super(ConvsLayer,self).__init__()   #继承卷积层对象的内容进行继承
        
        self.kernels = configs.kernels   #核的大小分别为 [13,15,17]
        hidden_channels = configs.cnn_chanel   #通过计算得到也就是输出
        in_channel = 1  #文本信息输入的通道数就为1
        features_L = configs.max_sequence_length  #序列最大的长度为500
        seq_dim = configs.seq_dim  #序列的维度为20
        dssp_dim = configs.dssp_dim  #dssp的维度为9
        pssm_dim = configs.pssm_dim #dssp的维度为20
        W_size = seq_dim + dssp_dim + pssm_dim #W_size = 49

        padding1 = (self.kernels[0]-1)//2 #避免数据丢失，在stride=1的条件下使得输入=输出
        padding2 = (self.kernels[1]-1)//2
        padding3 = (self.kernels[2]-1)//2
        self.conv1 = nn.Sequential()  #以字典形式按顺序添加神经网络，如下，"conv1",nn.conv1这样的形式
        self.conv1.add_module("conv1",
            nn.Conv2d(in_channel, hidden_channels,   #in_channel = 1,hidden_channels=228,有228个卷积核，输出位500*49*228
            padding=(padding1,0),
            kernel_size=(self.kernels[0],W_size))) #卷积核的大小为13*49
        self.conv1.add_module("ReLU",nn.PReLU()) # 增加激活函数： 增加网络的非线性分割能力
        self.conv1.add_module("pooling1",nn.MaxPool2d(kernel_size=(features_L,1),stride=1))
        #池化层主要的作用是特征提取(即删掉一些特征),进一步减少参数数量
        #hidden_channels理解为滤波器的个数或者通道数，就会得到这么多个特征图，通过maxpool2d之后就剩下一个值
        self.conv2 = nn.Sequential()
        self.conv2.add_module("conv2",
            nn.Conv2d(in_channel, hidden_channels,
            padding=(padding2,0),
            kernel_size=(self.kernels[1],W_size))) #(32,228,500,1)
        self.conv2.add_module("ReLU",nn.ReLU())
        self.conv2.add_module("pooling2",nn.MaxPool2d(kernel_size=(features_L,1),stride=1))
        
        self.conv3 = nn.Sequential()
        self.conv3.add_module("conv3",
            nn.Conv2d(in_channel, hidden_channels,
            padding=(padding3,0),
            kernel_size=(self.kernels[2],W_size)))
        self.conv3.add_module("ReLU",nn.ReLU())
        self.conv3.add_module("pooling3",nn.MaxPool2d(kernel_size=(features_L,1),stride=1))

    
    def forward(self,x):  #计算全局特征  前向的过程中定义输入

        features1 = self.conv1(x)  #通过卷积层获得全局的三个特征,输入的维度是(32,1,500,49)
        features2 = self.conv2(x)
        features3 = self.conv3(x)
        features = t.cat((features1,features2,features3),1) #按列进行拼接1维度。
        shapes = features.data.shape  # 得到的是特征的维度
        features = features.view(shapes[0],shapes[1]*shapes[2]*shapes[3])#view在torch里面的作用是重塑形状,相当于reshape
        #[batch_size,height,width, channels]  //图像张量
        #[个数，高度，宽度，通道数]
        #[height,width,input_channels,output_channels]   //滤波器张量
        #[卷积核高度，卷积核宽度，图像通道数，卷积核个数]  //第三维input_channels为input张量的第四维。
        #张量和数组格式相同：（a,b,c,d）a为个数，b为深度，c为行，d为列，深度表示1个张量中里面的个数，3维不存在个数的概念因为只有一个
        return features




#计算数据的预处理和计算全局特征

class DeepPPI(BasicModule):
    def __init__(self,class_nums,window_size,ratio=None):
        super(DeepPPI,self).__init__()
        global configs
        configs.kernels = [13, 15, 17]
        self.dropout = configs.dropout = 0.2  #采用drop来防止过拟合的现象,造Dropout方法，在每次训练过程中都随机“掐死”百分之二十的神经元，防止过拟合。

        seq_dim = configs.seq_dim*configs.max_sequence_length #20*500
        
        #将通过embeding的方法将稀疏向量转为密度向量
        self.seq_layers = nn.Sequential()
        self.seq_layers.add_module("seq_embedding_layer",
        nn.Linear(seq_dim,seq_dim))  #输入和输出的维度都是10000
        self.seq_layers.add_module("seq_embedding_ReLU",
        nn.ReLU())


        seq_dim = configs.seq_dim
        dssp_dim = configs.dssp_dim
        pssm_dim = configs.pssm_dim
        local_dim = (window_size*2+1)*(pssm_dim+dssp_dim+seq_dim) #就是（2n+1）*49=343
        if ratio:
            configs.cnn_chanel = (local_dim*int(ratio[0]))//(int(ratio[1])*3)  #ratio[0]=2,ratio[1]=1  取整之后的结果
        input_dim = configs.cnn_chanel*3+local_dim  #通道的大小为228，一共有三个
        #输入分类器里面数据维度是1027 = int(343*2/3*1)*3+343
        #cnn_chanel可能是想多了作者就是这么能定义计算的

        self.multi_CNN = nn.Sequential()
        self.multi_CNN.add_module("layer_convs",
                               ConvsLayer())  #再进行计算全局的特征

        
        #最后定义分类的网络
        self.DNN1 = nn.Sequential()
        self.DNN1.add_module("DNN_layer1",
                            nn.Linear(input_dim,1024))
        self.DNN1.add_module("ReLU1",
                            nn.ReLU())
        #self.dropout_layer = nn.Dropout(self.dropout)
        self.DNN2 = nn.Sequential()  #pytorch通过nn.Linear搭建全连接层，tf通过dense搭建全连接层
        self.DNN2.add_module("DNN_layer2",
                            nn.Linear(1024,256))  #设置网络中的全连接层的，需要注意的是全连接层的输入与输出都是二维张量
        self.DNN2.add_module("ReLU2",
                            nn.ReLU())

        # 代码注释
        self.outLayer = nn.Sequential(
            nn.Linear(256, class_nums),#因为class_nums=1所以最终的数量为1
            nn.Sigmoid())

    #提取局部的特征  从这里开始看0
    def forward(self,seq,dssp,pssm,local_features):
        shapes = seq.data.shape  #shape指的是序列的shape[32,1,500,20]
        features = seq.view(shapes[0],shapes[1]*shapes[2]*shapes[3]) #将4维的数据转为2维数据，32行500*20列的数据，将四维数据转为2维数据
        features = self.seq_layers(features)  #将稀疏矩阵转为密集矩阵，放入全连接层
        features = features.view(shapes[0],shapes[1],shapes[2],shapes[3]) #再把序列的特征转为四维，因为pssm和dssp是四维的

        features = t.cat((features,dssp,pssm),3)  #将序列和其他两个特征进行拼接
        features = self.multi_CNN(features)  #得到的特征是二维的
        features = t.cat((features, local_features), 1)  #将局部特征和全局特征连接在一起沿着水平方向

        #代码注释
        features = self.DNN1(features)
        #features =self.dropout_layer(features)
        features = self.DNN2(features)
        features = self.outLayer(features)
        # 代码注释

        return features