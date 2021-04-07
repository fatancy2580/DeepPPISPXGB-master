#-*- encoding:utf8 -*-



import os
import time
import pickle
import torch as t
import numpy as np
from torch.utils import data


#my lib
from utils.config import DefaultConfig 




class dataSet(data.Dataset):
    def __init__(self,window_size,sequences_file=None,pssm_file=None, dssp_file=None, label_file=None, protein_list_file=None):
        super(dataSet,self).__init__()  #继承dataSet的方法


        #读取五个文件
        self.all_sequences = []
        for seq_file in sequences_file: #读取sequences_file里面的内容,因为有四个文件所以需要一个个遍历
            with open(seq_file,"rb") as fp_seq:
               temp_seq  = pickle.load(fp_seq)
            self.all_sequences.extend(temp_seq)

        self.all_pssm = []
        for pm_file in pssm_file: 
            with open(pm_file,"rb") as fp_pssm:
                temp_pssm = pickle.load(fp_pssm)
            self.all_pssm.extend(temp_pssm)

        self.all_dssp = []
        for dp_file in dssp_file: 
            with open(dp_file,"rb") as fp_dssp:
                temp_dssp  = pickle.load(fp_dssp)
            self.all_dssp.extend(temp_dssp)

        self.all_label = []
        for lab_file in label_file: 
            with open(lab_file, "rb") as fp_label:
                temp_label = pickle.load(fp_label)
            self.all_label.extend(temp_label)

        with open(protein_list_file, "rb") as list_label:
            self.protein_list = pickle.load(list_label)

         

        self.Config = DefaultConfig()
        self.max_seq_len = self.Config.max_sequence_length
        self.window_size = window_size

        

    def __getitem__(self,index):   #让对象实现迭代功能
        
        count,id_idx,ii,dset,protein_id,seq_length = self.protein_list[index] #protein_list保存的是所有的数据
        window_size = self.window_size
        id_idx = int(id_idx)
        win_start = ii - window_size   #开始的索引
        win_end = ii + window_size     #终止的索引
        seq_length = int(seq_length)
        label_idx = (win_start+win_end)//2   #标签的索引为中间位置


        #全局特征的生成办法
        all_seq_features = []
        seq_len = 0
        for idx in self.all_sequences[id_idx][:self.max_seq_len]:  #最后蛋白质序列通过embedding转为稀疏向量
            acid_one_hot = [0 for i in range(20)] #长度为20的额全零列表
            acid_one_hot[idx] = 1  #序列采用one-hot编码
            all_seq_features.append(acid_one_hot)
            seq_len += 1
        while seq_len<self.max_seq_len:
            acid_one_hot = [0 for i in range(20)]
            all_seq_features.append(acid_one_hot)
            seq_len += 1

        all_pssm_features = self.all_pssm[id_idx][:self.max_seq_len]  #pssm采用密度向量
        seq_len = len(all_pssm_features)
        while seq_len<self.max_seq_len:
            zero_vector = [0 for i in range(20)]
            all_pssm_features.append(zero_vector)
            seq_len += 1

        all_dssp_features = self.all_dssp[id_idx][:self.max_seq_len] #为什么没用ont-hot因为本来就是one_hot
        seq_len = len(all_dssp_features)
        while seq_len<self.max_seq_len:
            zero_vector = [0 for i in range(9)]  #长度不够后面用全零数组添加
            all_dssp_features.append(zero_vector)
            seq_len += 1
        
        #其作用就是生成局部特征
        local_features = []
        labels = []
        while win_start<0:  #前面和后面两部分进行填充直到第window_size为止
            data = []
            acid_one_hot = [0 for i in range(20)]
            data.extend(acid_one_hot)

            pssm_zero_vector = [0 for i in range(20)]
            data.extend(pssm_zero_vector)

            dssp_zero_vector = [0 for i in range(9)]
            data.extend(dssp_zero_vector)

            local_features.extend(data)
            win_start += 1
       
        valid_end = min(win_end,seq_length-1) #最后window_size部分取其中的最小值
        while win_start<=valid_end:
            data = []
            idx = self.all_sequences[id_idx][win_start]

            acid_one_hot = [0 for i in range(20)]
            acid_one_hot[idx] = 1
            data.extend(acid_one_hot)


            pssm_val = self.all_pssm[id_idx][win_start]
            data.extend(pssm_val)

            try:
                dssp_val = self.all_dssp[id_idx][win_start]
            except:
                dssp_val = [0 for i in range(9)]
            data.extend(dssp_val)

            local_features.extend(data)
            win_start += 1

        while win_start<=win_end: #
            data = []
            acid_one_hot = [0 for i in range(20)]
            data.extend(acid_one_hot)

            pssm_zero_vector = [0 for i in range(20)]
            data.extend(pssm_zero_vector)

            dssp_zero_vector = [0 for i in range(9)]
            data.extend(dssp_zero_vector)

            local_features.extend(data)
            win_start += 1


        label = self.all_label[id_idx][label_idx] #第id_idx序列的第label_idex个标签
        label = np.array(label,dtype=np.float32)

        all_seq_features = np.stack(all_seq_features)#将得到的特征默认横向堆叠在一起
        all_seq_features = all_seq_features[np.newaxis,:,:] #np.newaxis表示增加一个坐标轴，二维变三维
        all_pssm_features = np.stack(all_pssm_features)#将得到的数组进行组合
        all_pssm_features = all_pssm_features[np.newaxis,:,:]

        all_dssp_features = np.stack(all_dssp_features)
        all_dssp_features = all_dssp_features[np.newaxis,:,:]
        local_features = np.stack(local_features)

        
        #这里得到的数据是三维的
        return all_seq_features,all_pssm_features,all_dssp_features,local_features,label
        # return  local_features, label


                

    def __len__(self):   #其作用是确定__getitem__的长度

        return len(self.protein_list)