import numpy as np
import pandas
import pandas as pd
import re
from tqdm import tqdm
strr=np.load(r"2023_3_3.npz")['x']
# yy0=pd.read_excel('2023_3_3.xlsx')
fin=[]
y=[]

for fff in tqdm(range(len(strr))):

    # with open(path+'\\'+file[fff],'r') as f:
    #     x=f.readlines()
    # print(x)
    x=strr[fff].split('\n')[:-1]
    num=re.findall('\d+',x[6])
    # try:
    #     if int(num[0])+int(num[1])+int(num[2])>36:
    #         continue
    # except:
    #     print(file[fff])
    #     os.remove(path+'\\'+file[fff])
    #     continue
    name=re.findall('[A-Za-z]+',x[0])
    a=[float(m) for m in x[2].replace('\n','').split(' ') if m !='']
    b=[float(m) for m in x[3].replace('\n','').split(' ') if m !='']
    c=[float(m) for m in x[4].replace('\n','').split(' ') if m !='']

    # mx=np.max([a,b,c])+0.0001
    mn = np.min([a, b, c])


    x0=re.findall('\d+',x[0])


    # print(num,name)
    data=x[8:]
    ache=[]

    for ll in data:
        tt=re.findall('\d+.\d+',ll)
        ache.append([float(tt[xx]) for xx in range(3)])
    # print(ache)

    nnum=sum([int(_) for _ in num])
    ff = 128 // nnum
    bb = 128 % nnum
    da=[]
    arg=[]
    ii=0
    ele=0
    for sss in num:
        ele+=1
        for ss in range(int(sss)):
            da.extend(ff*[ache[ii]] )
            if ele==1:
                arg.extend(ff*[[1,0,0]])
            elif ele==2:
                arg.extend(ff*[[0, 1, 0]] )
            elif ele==3:
                arg.extend(ff*[[0, 0, 1]] )
            ii+=1
    arg.extend([arg[-1] ]* bb)
    da.extend([da[-1]] * bb)

    a1=[np.sqrt(_[0]**2+_[1]**2+_[2]**2) for _ in [a,b,c]]
    a2=np.around([np.arccos(np.dot(b,c)/(a1[1]*a1[2]))/np.pi ,np.arccos(np.dot(a,c)/(a1[0]*a1[2]))/np.pi,np.arccos(np.dot(b,a)/(a1[1]*a1[0]))/np.pi],decimals=4)
    # print(a2)
    las=64*[a2.tolist()]
    las.extend(64*[np.divide(a1,15).tolist()])
    # print(las)
    fin.append([da,arg,las])
    # y.append(name)
    # print(np.array([pp0,pp1,pp2,21*[a]+21*[b]+22*[c]]).shape)
    # print(np.array(fin))

np.savez_compressed('6aaaaa.npz',x=fin)

da=np.load(r'6aaaaa.npz')['x']
