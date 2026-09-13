
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from model20.network import ProGAN
from model20.configuration import hparams
import torchvision.utils as vutils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint = './Models/Final Full Model.pth'
generator = ProGAN(hparams).load_model(checkpoint).gen_shadow.to(device)
generator.eval()
ProGanModel = ProGAN(hparams)
ProGanModel = ProGanModel.to(device)
num_samples = 16
z_dim = 512
test = ProGanModel(torch.rand(num_samples, z_dim, device=device)).detach().cpu()
fig = plt.figure(figsize=(20,20))
plt.imshow(np.transpose(vutils.make_grid(test, normalize=True), (1, 2, 0)))



#def truncated_normal(self, t, mean=0.0, std=1.0, limit=0.7):
#    while True:
#        cond = torch.logical_or(t < -limit, t > limit)
#        if not torch.sum(cond):
#            break
#        t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape, device=t.device), mean=mean, std=std), t)
#    return t

#def get_z():
#    z_sample = self.truncated_normal(torch.randn([self.z_samples, self.z_dim], device=self.device))
#    z = nn.Parameter(start_z).detach()

#def generate():
#    generates = generator(get_z())
#    generates = torch.clamp(generates, 0, 1)
#    return generates

#generate()

