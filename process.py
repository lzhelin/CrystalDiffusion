from sklearn.cluster import KMeans
import numpy as np
from sklearn.cluster import DBSCAN
import pandas as pd




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

def process(path,sh,data,el):#path，数量，数据，元素
    for s in range(sh):
        # print(s)
        # print(data[s])
        st = ''
        abc = data[s][2]
        # print(data[s][1])
        pos = []
        # print(abc)
        a0 = []
        num = []
        # print(abc)
        # print(np.average(abc[:64],axis=0),np.average(abc[64:],axis=0))
        cd = np.average(abc[64:], axis=0)
        # a0.append(np.average(abc[43:85],axis=0))
        jd = np.around(np.average(abc[:64] * np.pi, axis=0), decimals=4)
        alpha, beta, gamma = jd
        a, b, c = cd
        print(jd / np.pi, cd)
        if alpha > 2 * np.pi / 3 and beta > 2 * np.pi / 3 and gamma > 2 * np.pi / 3:
            print('error', s)
            continue
        # print(cd,jd)
        a0.append([a, 0, 0])
        a0.append([b * np.cos(gamma), b * np.sin(gamma), 0])
        e = c * (np.sqrt(
            1 + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma) - np.cos(alpha) ** 2 - np.cos(beta) ** 2 - np.cos(
                gamma) ** 2) / np.sin(gamma))
        vc = [c * np.cos(beta), c * ((np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)), c * (np.sqrt(
            1 + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma) - np.cos(alpha) ** 2 - np.cos(beta) ** 2 - np.cos(
                gamma) ** 2) / np.sin(gamma))]
        a0.append(vc)
        # print(a0)
        # print(a,b,c)
        arg = np.argmax(data[s][1], axis=1)
        # mask=np.ma.masked_where(arg==0,data[s][0])

        a, b, c = [], [], []
        for i in range(128):
            if arg[i] == 0:
                a.append(data[s][0][i].tolist())
            elif arg[i] == 1:
                b.append(data[s][0][i].tolist())
            elif arg[i] == 2:
                c.append(data[s][0][i].tolist())
        # print(c)
        # print(a)
        model = DBSCAN(eps=0.2, min_samples=6).fit(a)
        # print(model.labels_)

        cls = pd.Series(model.labels_).value_counts().to_dict()
        num.append(len(cls))
        if len(cls) == 1:
            pos.append(np.average(a, axis=0).tolist())
        else:
            km = KMeans(n_clusters=(len(cls)))
            _ = km.fit_transform(a)
            for p in km.cluster_centers_.tolist():
                pos.append(p)
        if b != []:
            model = DBSCAN(eps=0.3, min_samples=6).fit(b)
            # print(model.labels_)

            cls = pd.Series(model.labels_).value_counts().to_dict()
            num.append(len(cls))
            if len(cls) == 1:
                pos.append(np.average(b, axis=0).tolist())
            else:
                km = KMeans(n_clusters=(len(cls)))
                _ = km.fit_transform(b)
                for p in km.cluster_centers_.tolist():
                    pos.append(p)
        else:
            num.append(0)
        if c != []:
            model = DBSCAN(eps=0.3, min_samples=6).fit(c)
            # print(model.labels_)
            cls = pd.Series(model.labels_).value_counts().to_dict()
            # print(cls)
            num.append(len(cls))
            # print(cls.keys())
            if list(cls.keys())[0] == -1:
                num.append(0)

            elif len(cls) == 1:
                pos.append(np.average(c, axis=0).tolist())

            else:
                km = KMeans(n_clusters=(len(cls)))
                _ = km.fit_transform(c)
                for p in km.cluster_centers_.tolist():
                    pos.append(p)
        else:
            num.append(0)
        # print(pos)

        # print(pos)
        # print(arg)
        st += '{}\n15.0000\n'.format(s)
        for sss in range(3):
            print(a0)
            st += '     {} {} {}\n'.format(str(round(a0[sss][0] * 100) / 100), str(round(a0[sss][1] * 100) / 100),
                                           str(round(a0[sss][2] * 100) / 100))
        st += '   {}   {}   {}\n'.format(el[s][0], el[s][1], el[s][2])
        st += '     {}     {}     {}\nDirect\n'.format(str(num[0]), str(num[1]), str(num[2]))
        for i in pos:
            # print(i)

            st += '  {}  {}  {}\n'.format(str(round(i[0] * 100) / 100), str(round(i[1] * 100) / 100),
                                          str(round(i[2] * 100) / 100))

        with open(r'{}_{}.vasp'.format(path,sh), 'w+') as f:
            f.writelines(st)