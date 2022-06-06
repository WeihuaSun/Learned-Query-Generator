from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os 
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pack_sequence,pad_packed_sequence
from dataloader import loadData
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
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
lambda_gp = 10


data,lens,alltokens = loadData()  
dataset = TensorDataset(data)   
dataloader = DataLoader(dataset,batch_size=opt.batch_size)    

SEQLEN = data.shape[1]
TOKENNUM = len(alltokens)


lens = []
class Generator(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers):
        super(Generator, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True)
        self.softmax = nn.LogSoftmax(dim=2)
    def forward(self,x):
        #x=pack_padded_sequence(x,lengths=lengths,batch_first=True,enforce_sorted=False)
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size,seq_length,hidden_size)
        out = self.softmax(out)
        #out = pad_packed_sequence(sequence=out,batch_first=True,padding_value=0.0)
        return out
    
        
class Discriminator(nn.Module):
    def __init__(self,hidden_size,num_layers,embedding_size,embedding_dim):
        super(Discriminator,self).__init__()
        self.embedding = nn.Embedding(embedding_size,embedding_dim,0)
        #nn.init.xavier_uniform_(self.embedding.weight)
        self.input_size = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.input_size,hidden_size,num_layers,batch_first=True)
        self.liner = nn.Linear(hidden_size,1)
    def forward(self,x):
        #with torch.backends.cudnn.flags(enabled=False):
            #x.shape[batch_size,seq_len,fsize]
            x = self.embedding(x)
            h0 = torch.ones(self.num_layers, x.size(0), self.hidden_size).to(device) 
            c0 = torch.ones(self.num_layers, x.size(0), self.hidden_size).to(device)
            out, _ = self.lstm(x, (h0, c0))# out: tensor of shape (batch_size, seq_length, hidden_size)
            out = self.liner(out[:,-1,:])# out.shape: [batch_size, 1]
            return out


from torch.autograd import Variable
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
"""
def compute_gradient_penalty(D, real_samples, fake_samples):
    with torch.no_grad():
        alpha = torch.rand((real_samples.size(0),1)).to(device)
        # Get random interpolation between real and fake samples
        gradlist = []
        delta = 1
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).int()#输入
        for x_dim in range(interpolates.size(1)):#for each dim
            d_interpolates = interpolates.clone()
            d_interpolates[:,x_dim] += delta
            grads = D(d_interpolates)-D(interpolates)
            gradlist.append(torch.pow(grads,2).squeeze())
        grads = torch.t(torch.stack(gradlist))
        grads = torch.sqrt(torch.sum(grads,dim=1)).unsqueeze(dim=-1)
        gradient_penalty = torch.mean(torch.pow((grads-1),2))
        return gradient_penalty
"""
def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    with torch.no_grad():
        alpha = torch.rand((real_samples.size(0),1)).to(device)
        beta = torch.rand((real_samples.size(0),1)).to(device)
        # Get random interpolation between real and fake samples
        interpolates_a = (alpha * real_samples + ((1 - alpha) * fake_samples)).int()
        interpolates_b = (beta * real_samples + ((1 - beta) * fake_samples)).int()
        d_interpolates_a = D(interpolates_a)
        d_interpolates_b = D(interpolates_b)
        d_abs = torch.abs(d_interpolates_a - d_interpolates_b)
        d_sub = ((interpolates_a.float()-interpolates_b.float()).norm(2,dim=1)).unsqueeze(dim=-1)
        gradient_penalty = torch.mean(torch.pow((d_abs.float()/d_sub-1.),2))
        return gradient_penalty
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

# Loss weight for gradient penalty



# Initialize generator and discriminator
generator = Generator(opt.latent_dim,TOKENNUM,2).to(device)
#def __init__(self,hidden_size,num_layers,embedding_size,embedding_dim):
discriminator = Discriminator(D_HIDDEN_SIZE,2,TOKENNUM,EMBEDDING_DIM).to(device)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

            
        
   
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

        real_validity = discriminator(real.int())
        fake_validity = discriminator(profake)
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real.data, profake.data)
        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        if i % opt.n_critic == 0:

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
            print(gradient_penalty)
    print(sample()[0])

            
            
             

                
                
    