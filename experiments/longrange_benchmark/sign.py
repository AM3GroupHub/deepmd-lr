"""Is the global charge sign a genuine degeneracy of the objective?"""
import numpy as np, torch
from deepmd_lr.pme import PMEEnergy
from deepmd.pt_expt.utils.env import DEVICE
DT=torch.float64
d=np.load('data/valid/set.000/coord.npy'); atype=np.loadtxt('data/valid/type.raw',dtype=int)
n=len(atype); box=14.0
coord=torch.tensor(d[:4].reshape(4,n,3),dtype=DT,device=DEVICE)
cell=(torch.eye(3,dtype=DT,device=DEVICE)*box).expand(4,3,3)
q=torch.tensor(np.where(atype==0,1.0,-1.0),dtype=DT,device=DEVICE).expand(4,n)
term=PMEEnergy(rcut=6.0,alpha=0.5,mesh_dimensions=(24,24,24))
e_pos=term(coord,q,cell).sum(dim=(1,2))
e_neg=term(coord,-q,cell).sum(dim=(1,2))
print('E(+q) :',[round(float(x),6) for x in e_pos])
print('E(-q) :',[round(float(x),6) for x in e_neg])
print('max |E(q) - E(-q)| =', float((e_pos-e_neg).abs().max()))
print()
print('Coulomb energy is bilinear in charge: q_i q_j is unchanged by q -> -q,')
print('so for a neutral cell the global sign is unobservable from E and F.')
