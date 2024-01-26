from PCCD.Unet import Unet
from PCCD.DDPM import GaussianDiffusion
import torch
from PCCD.process import *
import numpy as np

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



    def get_key(val):
        for key, value in m.items():
            if val == value:
                return key



    sh = 1
    image_classes = torch.Tensor([5]).to(torch.long).cuda()

    iss0 = torch.Tensor([1]).to(torch.long).cuda()
    ee= torch.Tensor([[m['Ca'],m['Mg'],m['O']]]).to(torch.long).cuda()
    el = [['Ca','Mg','O']]




    sampled_images = diffusion.sample(
        classes=image_classes,
        e=ee,
        iss=iss0,
        cond_scale=3.
    )

    data = sampled_images.to('cpu').detach().numpy()
    process('./', sh, data, el)
