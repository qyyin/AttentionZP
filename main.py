#coding=utf8
import os
import sys
import re
import argparse
import math
import timeit
import numpy
import random
import cPickle
sys.setrecursionlimit(1000000)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
import torchvision.transforms as T
from torch.optim import lr_scheduler

from conf import *
import utils
from data_generater import *
from net import *

print >> sys.stderr, "PID", os.getpid()
random.seed(0)
numpy.random.seed(0)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)
torch.cuda.set_device(args.gpu)

def net_copy(net,copy_from_net):
    mcp = list(net.parameters())
    mp = list(copy_from_net.parameters())
    n = len(mcp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:]

def get_predict(data,t):
    predict = []
    for result,output in data:
        max_index = -1
        for i in range(len(output)):
            if output[i] > t:
                max_index = i
        predict.append(result[max_index])
    return predict

def get_predict_max(data):
    predict = []
    for result,output in data:
        max_index = -1
        max_pro = 0.0
        for i in range(len(output)):
            if output[i][1] > max_pro:
                max_index = i
                max_pro = output[i][1]
        predict.append(result[max_index])
    return predict
 
 
def get_evaluate(data,overall=1713.0):
    best_result = {}
    best_result["hits"] = 0
    predict = get_predict_max(data)
    result = evaluate(predict,overall)
    if result["hits"] > best_result["hits"]:
        best_result = result
    return best_result

def evaluate(predict,overall):
    result = {}
    result["hits"] = sum(predict)
    result["performance"] = sum(predict)/overall
    return result

MAX = 2

def main():

    read_f = file("./data/train_data","rb")
    train_generater = cPickle.load(read_f)
    read_f.close()
    read_f = file("./data/emb","rb")
    embedding_matrix,_,_ = cPickle.load(read_f)
    read_f.close()
    test_generater = DataGnerater("test",256)

    print "Building torch model"
    model = Network(nnargs["embedding_size"],nnargs["embedding_dimention"],embedding_matrix,nnargs["hidden_dimention"],2,nnargs["attention"]).cuda()

    this_lr = 0.003
    optimizer = optim.Adagrad(model.parameters(), lr=this_lr)
    best_result = {}
    best_result["hits"] = 0
    best_model = Network(nnargs["embedding_size"],nnargs["embedding_dimention"],embedding_matrix,nnargs["hidden_dimention"],2,nnargs["attention"]).cuda()
     
    for echo in range(nnargs["epoch"]):
        cost = 0.0
        print >> sys.stderr, "Begin epoch",echo
        for data in train_generater.generate_data(shuffle=True):
            output,output_softmax = model.forward(data,dropout=nnargs["dropout"])
            loss = F.cross_entropy(output,torch.tensor(data["result"]).type(torch.cuda.LongTensor))
            optimizer.zero_grad()
            cost += loss.item()
            loss.backward()
            optimizer.step()
        print >> sys.stderr, "End epoch",echo,"Cost:", cost
        predict = []
        for data in train_generater.generate_dev_data():
            output,output_softmax = model.forward(data)
            for s,e in data["start2end"]:
                if s == e:
                    continue
                predict.append((data["result"][s:e],output_softmax[s:e]))
        result = get_evaluate(predict,float(len(predict)))
        if result["hits"] > best_result["hits"]:
            best_result = result
            best_result["epoch"] = echo 
            net_copy(best_model,model)
        sys.stdout.flush()
    torch.save(best_model,"./model/best_model") 
    predict = []
    for data in test_generater.generate_data():
        output,output_softmax = best_model.forward(data)
        for s,e in data["start2end"]:
            if s == e:
                continue
            predict.append((data["result"][s:e],output_softmax[s:e]))
    result = get_evaluate(predict)
    print "dev:",best_result["performance"]
    print "test:",result["performance"]
 
if __name__ == "__main__":
    main()
