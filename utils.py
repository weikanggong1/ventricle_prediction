import numpy as np
import torch

def BWAS_correlation(fMRI_2D_1,fMRI_2D_2):
    # fMRI_2D_1 and fMRI_2D_2 are both time * voxel (t * p1 and t * p2) matrices
    # This function return the fisher z transformed correlation matrix (p1 * p2)
    
    fMRI_2D_1 = (fMRI_2D_1 - fMRI_2D_1.mean(axis=0)) / fMRI_2D_1.std(axis=0)
    fMRI_2D_2 = (fMRI_2D_2 - fMRI_2D_2.mean(axis=0)) / fMRI_2D_2.std(axis=0)
    r=np.dot(np.transpose(fMRI_2D_1),fMRI_2D_2) / fMRI_2D_1.shape[0]
    
    return r

class compute_loss_multi_conti(torch.nn.Module):
    def __init__(self, ntask=5):
        super(compute_loss_multi_conti, self).__init__()
        
        self.ntask = ntask
        self.sigma = torch.nn.Parameter(torch.zeros(ntask))

    def forward(self, pred, targets):
        
        precision = torch.exp(-self.sigma)

        targets = targets.contiguous()
        loss = 0.0
        for i in range(0, targets.shape[1]):
            loss = loss + precision[i] * torch.nn.functional.mse_loss(pred[:,i].contiguous(), targets[:,i].contiguous()) + self.sigma[i]
        
        return loss/targets.shape[1]



