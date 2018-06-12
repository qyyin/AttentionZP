# -*- coding: utf-8 -*-
import sys

import math
import random
import numpy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
import torch.autograd as autograd
import torchvision.transforms as T
import torch.optim as optim

from conf import *

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
random.seed(0)
numpy.random.seed(0)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)

class Network(nn.Module):
    def __init__(self, embedding_size, embedding_dimention, embedding_matrix, hidden_dimention, output_dimention,attention_d=2):
        super(Network,self).__init__()
        self.embedding_layer = nn.Embedding(embedding_size,embedding_dimention)
        self.embedding_layer.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.inpt_layer_zp_pre = nn.Linear(embedding_dimention,hidden_dimention)
        self.hidden_layer_zp_pre = nn.Linear(hidden_dimention,hidden_dimention,bias=False)
        self.inpt_layer_zp_post = nn.Linear(embedding_dimention,hidden_dimention)
        self.hidden_layer_zp_post = nn.Linear(hidden_dimention,hidden_dimention,bias=False)
        self.inpt_layer_np = nn.Linear(embedding_dimention,hidden_dimention)
        self.hidden_layer_np = nn.Linear(hidden_dimention,hidden_dimention)
        nh = hidden_dimention*2
        self.zp_pre_representation_layer = nn.Linear(hidden_dimention,nh)
        self.zp_post_representation_layer = nn.Linear(hidden_dimention,nh)
        self.np_representation_layer = nn.Linear(hidden_dimention,nh)
        self.nps_representation_layer = nn.Linear(hidden_dimention,nh)
        self.feature_representation_layer = nn.Linear(nnargs["feature_dimention"],nh)
        self.representation_hidden_layer = nn.Linear(hidden_dimention*2,hidden_dimention*2)
        self.output_layer = nn.Linear(hidden_dimention*2,output_dimention)
        self.hidden_size = hidden_dimention
        self.activate = nn.Tanh()
        self.selfAttentionA_pre = nn.Linear(hidden_dimention,128,bias=False)
        self.selfAttentionB_pre = nn.Linear(128,attention_d,bias=False)
        self.selfAttentionA_post = nn.Linear(hidden_dimention,128,bias=False)
        self.selfAttentionB_post = nn.Linear(128,attention_d,bias=False)
        self.Attention_np = nn.Linear(256,1)
        self.Attention_zp_post = nn.Linear(256,1,bias=False)
        self.Attention_zp_pre = nn.Linear(256,1,bias=False)
        self.selfAttentionA_np = nn.Linear(hidden_dimention,128,bias=False)
        self.selfAttentionB_np = nn.Linear(128,attention_d,bias=False)
    def forward_zp_pre(self, word_index, hiden_layer,dropout=0.0):
        dropout_layer = nn.Dropout(dropout)
        word_embedding = self.embedding_layer(word_index)#.view(-1,word_embedding_rep_dimention)
        word_embedding = dropout_layer(word_embedding)
        this_hidden = self.inpt_layer_zp_pre(word_embedding) + self.hidden_layer_zp_pre(hiden_layer)
        this_hidden = self.activate(this_hidden)
        this_hidden = dropout_layer(this_hidden)
        return this_hidden
    def forward_zp_post(self, word_index, hiden_layer,dropout=0.0):
        dropout_layer = nn.Dropout(dropout)
        word_embedding = self.embedding_layer(word_index)#.view(-1,word_embedding_rep_dimention)
        this_hidden = self.inpt_layer_zp_post(word_embedding) + self.hidden_layer_zp_post(hiden_layer)
        this_hidden = self.activate(this_hidden)
        this_hidden = dropout_layer(this_hidden)
        return this_hidden
    def forward_np(self, word_index, hiden_layer,dropout=0.0):
        dropout_layer = nn.Dropout(dropout)
        word_embedding = self.embedding_layer(word_index)
        this_hidden = self.inpt_layer_np(word_embedding) + self.hidden_layer_np(hiden_layer)
        this_hidden = self.activate(this_hidden)
        this_hidden = dropout_layer(this_hidden)
        return this_hidden
    def generate_score(self,zp_pre,zp_post,np,feature,dropout=0.0):
        dropout_layer = nn.Dropout(dropout)
        x = self.zp_pre_representation_layer(zp_pre) + self.zp_post_representation_layer(zp_post) + self.np_representation_layer(np)\
            + self.feature_representation_layer(feature) 
        x = self.activate(x)
        x = dropout_layer(x)
        x = self.representation_hidden_layer(x)
        x = self.activate(x)
        x = dropout_layer(x)
        x = self.output_layer(x)
        xs = F.softmax(x)
        return x,xs
    def initHidden(self,batch=1):
        return torch.tensor(numpy.zeros((batch, self.hidden_size))).type(torch.cuda.FloatTensor)
    def get_attention_pre(self,inpt):
        return self.selfAttentionB_pre(self.activate(self.selfAttentionA_pre(inpt)))
    def get_attention_post(self,inpt):
        return self.selfAttentionB_post(self.activate(self.selfAttentionA_post(inpt)))
    def forward(self,data,dropout=0.0):
        zp_reindex = torch.tensor(data["zp_reindex"]).type(torch.cuda.LongTensor)
        zp_pre = torch.tensor(data["zp_pre"]).type(torch.cuda.LongTensor)
        zp_pre_mask = torch.tensor(data["zp_pre_mask"]).type(torch.cuda.FloatTensor)
        zp_post = torch.tensor(data["zp_post"]).type(torch.cuda.LongTensor)
        zp_post_mask = torch.tensor(data["zp_post_mask"]).type(torch.cuda.FloatTensor)
        candi_reindex = torch.tensor(data["candi_reindex"]).type(torch.cuda.LongTensor)
        candi = torch.tensor(data["candi"]).type(torch.cuda.LongTensor)
        candi_mask = torch.tensor(data["candi_mask"]).type(torch.cuda.FloatTensor)
        feature = torch.tensor(data["fl"]).type(torch.cuda.FloatTensor)
        zp_pre = torch.transpose(zp_pre,0,1)
        mask_zp_pre = torch.transpose(zp_pre_mask,0,1)
        hidden_zp_pre = self.initHidden()
        hiddens_zp_pre = []
        for i in range(len(mask_zp_pre)):
            hidden_zp_pre = self.forward_zp_pre(zp_pre[i],hidden_zp_pre,dropout=dropout)*torch.transpose(mask_zp_pre[i:i+1],0,1)
            hiddens_zp_pre.append(hidden_zp_pre)
        hiddens_zp_pre = torch.cat(hiddens_zp_pre,1)
        hiddens_zp_pre = hiddens_zp_pre.view(-1,len(mask_zp_pre),nnargs["hidden_dimention"])
        pre_A = self.get_attention_pre(hiddens_zp_pre)
        pre_A = F.softmax(pre_A,1)
        average_results_pre = torch.matmul(torch.transpose(hiddens_zp_pre,1,2),pre_A)
        zp_pre_attention = torch.sum(average_results_pre,2)
        zp_pre_representation = zp_pre_attention[zp_reindex]
        zp_post = torch.transpose(zp_post,0,1)
        mask_zp_post = torch.transpose(zp_post_mask,0,1)
        hidden_zp_post = self.initHidden()
        hiddens_zp_post = []
        for i in range(len(mask_zp_post)):
            hidden_zp_post = self.forward_zp_post(zp_post[i],hidden_zp_post,dropout=dropout) *torch.transpose(mask_zp_post[i:i+1],0,1)
            hiddens_zp_post.append(hidden_zp_post)
        hiddens_zp_post = torch.cat(hiddens_zp_post,1)
        hiddens_zp_post = hiddens_zp_post.view(-1,len(mask_zp_post),nnargs["hidden_dimention"])
        post_A = self.get_attention_post(hiddens_zp_post)
        post_A = F.softmax(post_A,1)
        average_results_post = torch.matmul(torch.transpose(hiddens_zp_post,1,2),post_A)
        zp_post_attention = torch.sum(average_results_post,2)
        zp_post_representation = zp_post_attention[zp_reindex]
        candi = torch.transpose(candi,0,1)
        mask_candi = torch.transpose(candi_mask,0,1)
        hidden_candi = self.initHidden()
        hiddens_candi = []
        for i in range(len(mask_candi)):
            hidden_candi = self.forward_np(candi[i],hidden_candi,dropout=dropout)*torch.transpose(mask_candi[i:i+1],0,1)
            hiddens_candi.append(hidden_candi)
        hiddens_candi = torch.cat(hiddens_candi,1)
        hiddens_candi = hiddens_candi.view(-1,len(mask_candi),nnargs["hidden_dimention"])
        nps = []
        for npt,pret,postt in zip(hiddens_candi,zp_pre_representation,zp_post_representation):
            attention = F.softmax(torch.squeeze(self.activate(self.Attention_np(npt)+self.Attention_zp_post(postt)+self.Attention_zp_pre(pret))))
            average_np = torch.transpose(npt,0,1)*attention
            average_np = torch.sum(average_np,1,keepdim=True)
            nps.append(average_np)
        nps = torch.transpose(torch.cat(nps,1),0,1)
        candi_representation = nps[candi_reindex]
        output,softmax_out = self.generate_score(zp_pre_representation,zp_post_representation,candi_representation,feature)
        output = torch.squeeze(output)
        return output,softmax_out
