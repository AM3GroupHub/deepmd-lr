"""How much of the label is invisible to a strictly local 6 A model?"""
import numpy as np, torch
from deepmd.dpmodel.modifier.dipole_charge import ELECTROSTATIC_CONVERSION as KE
from generate_data import coulomb
DT=torch.float64; dev='cuda'
d=np.load('data/valid/set.000/coord.npy'); box=14.0; n=d.shape[1]//3
atype=np.loadtxt('data/valid/type.raw',dtype=int)
q=torch.tensor(np.where(atype==0,1.0,-1.0),dtype=DT,device=dev)
cell=torch.eye(3,dtype=DT,device=dev)*box
F_full=np.load('data/valid/set.000/force.npy')

def truncated_coulomb_force(pos,rcut):
    pos=pos.clone().requires_grad_(True)
    inv=torch.linalg.inv(cell)
    delta=pos[:,None,:]-pos[None,:,:]
    frac=delta@inv; delta=(frac-torch.round(frac))@cell
    r=torch.linalg.norm(delta,dim=-1)
    eye=torch.eye(n,dtype=torch.bool,device=dev)
    m=(r<rcut)&(~eye)
    safe=torch.where(m,r,torch.ones_like(r))
    e=0.5*((q[:,None]*q[None,:]*KE/safe)*m).sum()
    return -torch.autograd.grad(e,pos)[0]

diffs=[]
for i in range(40):
    pos=torch.tensor(d[i].reshape(n,3),dtype=DT,device=dev)
    p=pos.clone().requires_grad_(True)
    f_exact=-torch.autograd.grad(coulomb(p,q,cell,dev),p)[0]
    f_trunc=truncated_coulomb_force(pos,6.0)
    diffs.append(float(((f_exact-f_trunc)**2).mean().sqrt()))
tot=np.sqrt((F_full**2).mean()); beyond=np.mean(diffs)
print(f'total label force RMS         : {tot:.4f} eV/A')
print(f'Coulomb beyond 6 A, force RMS : {beyond:.4f} eV/A')
print(f'  -> {beyond/tot*100:.1f}% of the total force signal is outside the cutoff')
print(f'  -> that is the floor a strictly local 6 A model cannot go below')
