from PCCD.Unet import Unet
from PCCD.DDPM import GaussianDiffusion
import torch
from PCCD.process import *
import numpy as np
import numpy
xs = np.load('inv.npz', allow_pickle=True)
if __name__ == '__main__':
    # os.mkdir('save')
    # os.mkdir('save1')
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8, 16),
        num_classes=10,
        cond_drop_prob=0.2
    ).cuda()
    ptt = torch.load(r'cond.pkl')
    model.load_state_dict(ptt)
    diffusion = GaussianDiffusion(
        model,
        image_size=128,
        timesteps=1000
    ).cuda()

    name = np.load('mp.npy')
    fm = ['AFM' 'FM' 'FiM' 'NM' 'Unknown']


    def get_key(val):
        for key, value in m.items():
            if val == value:
                return key


    for ci in range(1):

        sh = 16
        # torch.save(model.state_dict(), r'/kaggle/working/pt/cond_{}.pkl'.format(epoch))
        # os.mkdir('32')
        image_classes = torch.Tensor(xs['classes'][128 * ci:128 * (ci + 1)]).to(torch.long).cuda()
        # print([random.randint(1,3)])
        iss0 = torch.Tensor(xs['iss'][128 * ci:128 * (ci + 1)]).to(torch.long).cuda()
        elem = []
        el = []
        sc = xs['e'][ci * 128:(ci + 1) * 128]
        for ks in sc:
            xsx = []
            for j in ks:
                xsx.append(get_key(j))

            el.append(xsx)
        print(el)
        ee = torch.LongTensor(sc).cuda()
        print(ci)

        sampled_images = diffusion.sample(
            classes=image_classes,
            e=ee,
            iss=iss0,
            cond_scale=3.
        )

        data = sampled_images.to('cpu').detach().numpy()
        process(ci, sh, data, el)