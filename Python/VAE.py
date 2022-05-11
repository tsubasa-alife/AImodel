import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch import optim

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class VAE(nn.Module):
    def __init__(self,x_dim,z_dim):
        super(VAE,self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.x2mu = nn.Linear(x_dim,z_dim)
        self.x2log_var = nn.Linear(x_dim,z_dim)
        self.z2output = nn.Linear(z_dim,x_dim)

    def encoder(self,x):
        mu = self.x2mu(x)
        log_var = self.x2log_var(x)
        return mu,log_var

    def sample_z(self,mu,log_var):
        epsilon = torch.randn(mu.shape)
        return mu + epsilon * torch.exp(0.5 * log_var)

    def decoder(self,z):
        output = torch.tanh(self.z2output(z))
        return output

    def forward(self,x):
        mu,log_var = self.encoder(x)
        z = self.sample_z(mu,log_var)
        output = self.decoder(z)
        rec_loss = 0.5 * torch.sum((output - x)**2)
        kld = (- 0.5) * torch.sum(1 + log_var - mu**2 -torch.exp(log_var))
        elbo = rec_loss + kld
        return elbo,z,output

model = VAE(5,2)
optimizer = optim.Adam(model.parameters(),lr=0.001)
print(model)
model.train()
epochs = 10000
loss_list = []
x = torch.randn(1,5)
print("x:",x)
for i in range(epochs):
    loss,z,output = model.forward(x)
    #print("output:",output)
    #print("z:",z)
    print("loss:",loss)
    model.zero_grad()
    loss.backward()
    optimizer.step()
    loss_list.append(loss.detach().numpy())

fig = plt.figure()
plt.plot(loss_list)
plt.savefig("loss.png")