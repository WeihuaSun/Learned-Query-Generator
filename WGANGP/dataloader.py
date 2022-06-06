
from numpy import dtype
import torch
import re
from torch.utils.data import TensorDataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import csv
alltokens = []
filename = "./data/job-light.sql"

def readLines(filename):
    global alltokens
    alltokens.extend([",",";","=","!=",">=","<=",">","<"])
    alltokens = set(alltokens)
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    sentences = [[]for i in range(len(lines))]
    for i,line in enumerate(lines):
        tokens = re.split(r"\s|,|=|>|<|>=|<=|!=|;",line)
        for token in tokens:
            if token!="":
                alltokens.add(token)
                
        tokens = re.split(" ",line)
        for token in tokens:
            vm = re.search(r",|=|>|<|>=|<=|!=",token)
            if vm is not None:
                c = vm.group(0)
                temptokens = re.split(r",|=|>|<|>=|<=|!=",token)
                sentences[i].append(temptokens[0])
                sentences[i].append(c)
                sentences[i].append(temptokens[1])
            else:
                sentences[i].append(token)
        sentences[i][len(sentences[i])-1] = sentences[i][len(sentences[i])-1][0:-1]
        sentences[i].append(";")
    
    alltokens = list(alltokens)
    alltokens.sort()
    return alltokens,sentences
    
def tokenToIndex(token):
    return alltokens.index(token)

def tokenToTensor(token):
    tensor = torch.zeros(1, n_tokens)
    tensor[0][tokenToIndex(token)] = 1
    return tensor
def lineToTensor(line):
    return torch.Tensor([tokenToIndex(token) for token in line])
spath = "./data/process.pkl"
tpath = "./data/tokens.pkl"
def process():
    print("start process")
    print("read lines....")
    alltokens,sentences=readLines(filename)
    #lens = [len(data) for data in sentences]
    global n_tokens
    n_tokens = len(alltokens)
    print("line to tensor...")
    with open(spath,"ab+") as f:
        for i,s in enumerate (sentences):
            if i%100 ==0:
                print(f"files:{i}")
            sentence = lineToTensor(s)
            pickle.dump(sentence,f)
    
    #print("pad...")
    #sentences = pad_sequence(sentences,batch_first=True)
    with open(tpath,"ab+") as f:
        pickle.dump(alltokens,f)
    
    
def loadData():
    sentences=[]
    with open(spath,"rb") as f:
        for i in range(70):
            sentences.append(pickle.load(f))
    #sentences = torch.stack(sentences)
    print(len(sentences))
    sentences = pad_sequence(sentences,batch_first=True)
    with open(tpath,"rb+") as f:
        alltokens = pickle.load(f)
    return sentences,[],alltokens


#process()