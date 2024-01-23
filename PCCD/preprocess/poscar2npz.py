import numpy as np
import os
import re
import pandas as pd
from tqdm import tqdm
path='D:\poscar'
yy=[]
npz=[]
pdd=[]
y0=pd.read_excel('0.xlsx',sheet_name='Sheet1')
hh=y0.columns.values
print(hh)
h=y0.material_id
for ff in tqdm(y0.iterrows()):
    # print(ff[1])
    if ff[1].nelements>3 or ff[1].nsites>16:
        continue
    pdd.append(ff[1].values.tolist())
    text=''

    with open('D:\diffusion data\poscar2'+'\\'+ff[1][0],'r') as f:
        t=f.readlines()

    for i in t[:8]:
        text+=i
    for j in t[8:]:
        r=re.findall('[a-z]+',j,re.IGNORECASE)
        j1=re.sub(r[0],'',j)
        text+=j1


    npz.append(text)
    # print(npz)
    di={list(ff[1].values)[0]:list(ff[1].values)[1:]}
    yy.append(di)

tb = pd.DataFrame(pdd, columns=hh)
tb.to_excel('2023_3_3.xlsx')
np.savez_compressed('2023_3_3.npz',x=npz)
x=np.load('2023_3_3.npz',allow_pickle=True)
print(x['x'])