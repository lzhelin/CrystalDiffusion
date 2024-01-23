
import torch


from PCCD.Unet import Unet
from PCCD.DDPM import GaussianDiffusion
from PCCD.process import *



import os
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
try:
    os.mkdir(r'./pt1')
    os.mkdir(r'./res1')
except:
    print('you')
data=np.load('./6aaaaa.npz')
x=data['x']
x=torch.Tensor(x).to(torch.float).cuda()

y = pd.read_excel('./2023_3_3.xlsx', index_col=0)
enc = LabelEncoder()
y0 = enc.fit_transform(y.symbol)
y.symbol = y0
y0 = enc.fit_transform(y.crystal_system)
y.crystal_system = y0
y0 = enc.fit_transform(y.is_stable)
y.is_stable = y0
y0 = enc.fit_transform(y.is_gap_direct)
y.is_gap_direct = y0
y0 = enc.fit_transform(y.is_metal)
y.is_metal = y0
y0 = enc.fit_transform(y['Magnetic Ordering'])
y['Magnetic Ordering'] = y0

y0=y.band_gap.values
for i in range(len(y0)):
    if y0[i] >0 and y0[i]<1.5:
        y0[i]=5
    if y0[i]>=1.5:
        y0[i]=10
print(y0)
y.band_gap=y0

ele = y.elements.values
lab = []
wai = []
for ss in range(len(ele)):
    i = ele[ss]
    i = i.replace('Element', '')
    i = i.replace('[', '')
    i = i.replace(']', '')
    i = i.replace(' ', '')

    i = i.split(',')
    lleenn = len(i)
    if lleenn == 3:
        for kk in i:
            if kk not in lab:
                lab.append(kk)

        wai.append(i)
    if lleenn == 2:
        for kk in i:
            if kk not in lab:
                lab.append(kk)
        wai.append((i * 2)[:-1])

    if lleenn == 1:
        for kk in i:
            if kk not in lab:
                lab.append(kk)
        wai.append(i * 3)
m = {
    'Cs': 0, 'Rb': 1, 'K': 2, 'Na': 3, 'Li': 4, 'Ba': 5, 'Sr': 6, 'Ca': 7, 'Mg': 8, 'Ac': 9, 'Th': 10,
    'Pa': 11, 'U': 12, 'Np': 13, 'Pu': 14, 'La': 15, 'Ce': 16, 'Pr': 17, 'Nd': 18, 'Pm': 19, 'Sm': 20,
    'Eu': 21, 'Gd': 22, 'Tb': 23, 'Dy': 24, 'Ho': 24, 'Er': 25, 'Tm': 26, 'Yb': 27, 'Lu': 28,
    'Y': 29, 'Sc': 30, 'Hf': 31, 'Zr': 32, 'Ti': 33, 'Ta': 34, 'Nb': 35, 'V': 36, 'Cr': 37, 'Mn': 38, 'Fe': 39,
    'Co': 40, 'Ni': 41, 'Cu': 42, 'Zn': 43, 'W': 44, 'Mo': 45, 'Re': 46, 'Tc': 47, 'Os': 48, 'Ru': 49,
    'Ir': 50, 'Rh': 51, 'Pt': 52, 'Pd': 53, 'Au': 54, 'Ag': 55, 'Hg': 56, 'Cd': 57, 'B': 58, 'Tl': 59,
    'In': 60, 'Ga': 61, 'Al': 62, 'Be': 63, 'Pb': 64, 'Sn': 65, 'Ge': 66, 'Si': 67, 'C': 68, 'Bi': 69,
    'Sb': 70, 'As': 71, 'P': 72, 'N': 73, 'Te': 74, 'Se': 75, 'S': 76, 'O': 77, 'H': 78, 'I': 79, 'Br': 80,
    'Cl': 81, 'F': 82, 'Xe': 83, 'Kr': 84, 'Ar': 85, 'Ne': 86, 'He': 87
}
# print(wai)
# m=dict(zip(lab,[on for on in range(len(lab))]))
# print(m)
for liu in range(len(wai)):
    for nei in range(len(wai[liu])):
        wai[liu][nei] = m[wai[liu][nei]]



import torch
import torch

import os
import numpy as np

from torch import optim
from torch.utils.data import DataLoader,Dataset
try:
    # os.mkdir(r'./pt')
    os.mkdir(r'./res')
except:
    print('you')
print('start')
header=y.columns
class mydata1(Dataset):
    def __init__(self,data,y):
        self.data=data
        self.y=y
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, index):
        sample = self.data[index]
        return sample,self.y.iloc[index,:].to_list(),wai[index]


def train(data0, yy0, device='cpu', lr=1e-4, epochs=2000, image_size=(128, 3), channel=3):
    # setup_logging(args.run_name)
    global lsls
    # dataset = ImageFolder('.\car', transform)
    # dataset = mydata(data, transform)

    dataset = mydata1(data0, yy0)
    dataloader = DataLoader(dataset, batch_size=128)

    #     model = UNet(3,3,num_classes= len(m)).cuda()
    #     diffusion = Diffusion(image_size=image_size, device=device).cuda()

    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8,16),
        num_classes=10,
        cond_drop_prob=0.2
    )
    # pt=torch.load('pt/cond_700.pkl')
    # model.load_state_dict(pt)
    diffusion = GaussianDiffusion(
        model,
        image_size=128,
        timesteps=1000
    ).cuda()

    opt = optim.Adam(lr=lr, params=model.parameters())
    #     mse = nn.MSELoss()

    # logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    df = []
    asas=[]
    for epoch in range(epochs + 1):

        print('epoch:', epoch)
        ttt = 0
        for i, [images, y0, elem] in enumerate(dataloader):
            ttt += 1
            #             t = diffusion.sample_timesteps(images.shape[0])
            #             x_t, noise = diffusion.noise_images(images, t)
            elem = torch.stack(elem).to(torch.long).T.cuda()
            y0 = pd.DataFrame(y0).T
            y0.columns = header
            #             predicted_noise = model(x_t, t,elem)
            #             image_classes = torch.randint(0, num_classes, (8,)).cuda()
            sys = torch.stack(y0.band_gap.values.tolist()).to(torch.long).cuda()
            stab = torch.stack(y0['Magnetic Ordering'].values.tolist()).to(torch.long).cuda()
            #             print(sys.shape,elem.shape)
            loss = diffusion(images, classes=sys, iss=stab, e=elem)  # 损失函数
            df.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()



    torch.save(model.state_dict(), r'cond.pkl')







def launch():

    train(x, y, device='cuda:0')


if __name__ == '__main__':
    launch()
    pass
