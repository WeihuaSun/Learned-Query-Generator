from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os 
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
#from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pack_sequence,pad_packed_sequence
from dataloader import loadData
parser = argparse.ArgumentParser()
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--n_epochs", type=int, default=10000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=30, help="dimensionality of the latent space")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
opt = parser.parse_args()
print(opt)
cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda') if torch.cuda.is_available() else  torch.device('cpu')

EMBEDDING_DIM = 32
D_HIDDEN_SIZE = 32


data,lens,alltokens = loadData()  
dataset = TensorDataset(data)   
dataloader = DataLoader(dataset,batch_size=opt.batch_size)    

SEQLEN = data.shape[1]
TOKENNUM = len(alltokens)


lens = []
class Generator(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers):
        super(Generator, self).__init__()
        self.fc1 = nn.Sequential(
			nn.Linear(input_size*SEQLEN, input_size*SEQLEN),
			nn.BatchNorm1d(input_size*SEQLEN),
			nn.LeakyReLU(0.2, inplace=True),
		)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True)
        self.fc2 = nn.Sequential(
			nn.Linear(hidden_size*SEQLEN, hidden_size*SEQLEN),
			nn.BatchNorm1d(hidden_size*SEQLEN),
			nn.LeakyReLU(0.2, inplace=True),
		)
        self.softmax = nn.LogSoftmax(dim=2)
    def forward(self,x):
        #x=pack_padded_sequence(x,lengths=lengths,batch_first=True,enforce_sorted=False)
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Forward propagate LSTM
        out = x.view(x.size(0),self.input_size*SEQLEN)
        out = self.fc1(out)
        out = out.view(out.size(0),SEQLEN,self.input_size)
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size,seq_length,hidden_size)
        out = out.contiguous()
        out = out.view(out.size(0),self.hidden_size*out.size(1))
        out = self.fc2(out)
        out = out.view(out.size(0),SEQLEN,self.hidden_size)
        out = self.softmax(out)
        #out = pad_packed_sequence(sequence=out,batch_first=True,padding_value=0.0)
        return out
    
        
class Discriminator(nn.Module):
    def __init__(self,hidden_size,num_layers,embedding_size,embedding_dim):
        super(Discriminator,self).__init__()
        self.embedding = nn.Embedding(embedding_size,embedding_dim,0)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.input_size = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.input_size,hidden_size,num_layers,batch_first=True)
        self.liner = nn.Linear(hidden_size,1)
    def forward(self,x):
        #with torch.backends.cudnn.flags(enabled=False):
            #x.shape[batch_size,seq_len,fsize]
            x = self.embedding(x)
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            out, _ = self.lstm(x, (h0, c0))# out: tensor of shape (batch_size, seq_length, hidden_size)
            out = self.liner(out[:,-1,:])# out.shape: [batch_size, 1]
            return out



def sample(sample_num = 10):
    sqls = []
    z = torch.randn((sample_num,SEQLEN,opt.latent_dim), device=device)#(batch, seq, feature)
    with torch.no_grad():
        fakes =  generator(z)#(batch, seq, feature)
        for i,fake in enumerate (fakes):#(seqlen, feature)
            s=tensor2sql(fake)
            sqls.append(s)
    return sqls

def tensor2sql(fake):
    temp = []
    for word in fake:#feature
        index = word.argmax(dim=0).item()
        temp.append(alltokens[index])
    s = f" ".join(temp)
    return s
def indextosql(input):
    temp = []
    for index in input:
        temp.append(alltokens[index.item()])
    s = f" ".join(temp)
    return s
    
    
# Loss weight for gradient penalty



# Initialize generator and discriminator
generator = Generator(opt.latent_dim,TOKENNUM,2).to(device)
#def __init__(self,hidden_size,num_layers,embedding_size,embedding_dim):
discriminator = Discriminator(D_HIDDEN_SIZE,2,TOKENNUM,EMBEDDING_DIM).to(device)

optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)

            
        
   
for epoch in range(opt.n_epochs):
    for i, data in enumerate(dataloader):
        # Configure input
        real = data[0].to(device)
        #print(tensor2sql(real[0]))
        b_size = real.size(0) #(batch, seq, feature)
       
        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = torch.randn((b_size,SEQLEN,opt.latent_dim), device=device)#(batch, seq, feature)
        
        
        # Generate
        fake = generator(z)#(batch, seq, feature)
        profake = torch.argmax(fake,dim=2)
        #print(indextosql(profake[0]))
        #print(indextosql(real.int()[0]))
        real_validity = discriminator(real.int())
        fake_validity = discriminator(profake)
        # Gradient penalty

        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)

        d_loss.backward()
        optimizer_D.step()
        for p in discriminator.parameters():
                p.data.clamp_(-opt.clip_value, opt.clip_value)

        # Train the generator every n_critic steps
        if i % opt.n_critic == 0:
            optimizer_G.zero_grad()
            # -----------------
            #  Train Generator
            # -----------------
            # Generate a batch of images
            fake = generator(z)
            profake = torch.argmax(fake,dim=2)
            fake_validity = discriminator(profake)
            g_loss = -torch.mean(fake_validity)
            g_loss.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            ) 
    if epoch%100==0:
        print(sample()[0])

            
            
             

                
                
    