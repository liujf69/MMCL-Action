import torch
import torch.nn as nn
import torch.nn.functional as F

class Contrastive_loss(nn.Module):
    def __init__(self, tau = 0.4):
        super(Contrastive_loss, self).__init__()
        self.tau = tau
    
    def sim(self, z1, z2): 
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t()) 
    
    def semi_loss(self, z1, z2):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
    
    def forward(self, z1, z2, mean = True):
        l1 = self.semi_loss(z1, z2)
        l2 = self.semi_loss(z2, z1)
        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()
        return ret

class FAM_Aligh(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FAM_Aligh, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.mlp = nn.Sequential(
            nn.Linear(self.in_channel, self.out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_channel, self.out_channel)
        )
        self.loss = Contrastive_loss()
        
    def forward(self, ske_feature, rgb_feature):
        NM, C1, T, V = ske_feature.shape
        N, C2 = rgb_feature.shape
        ske_feature = ske_feature.mean(2).mean(2).reshape(N, 2, C1).mean(1) # global
        rgb_feature = self.mlp(rgb_feature) 
        loss = self.loss(ske_feature, rgb_feature)
        return loss, rgb_feature
    
