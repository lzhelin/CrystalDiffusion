# CrystalDiffusion

## Environment

You can run ```!pip install -r requirement.txt``` to install packages required.


## Usage

[npz2cloud.py](PCCD/preprocess/npz2cloud.py)and [npz2cloud.py](PCCD/preprocess/npz2cloud.py) in PCCD/preprocess is for data preprocessing.

run ```python train.py``` to train PCCD.

If you want use PCCD, you can run:

    python train.py

You can also use it as follows:
```python
from PCCD.Unet import Unet
from PCCD.DDPM import GaussianDiffusion
import torch
from PCCD.process import *
import numpy as np


sh = 1
image_classes = torch.Tensor([5]).to(torch.long).cuda()

iss0 = torch.Tensor().to(torch.long).cuda()
elem = torch.Tensor([[m['Ca'],m['Mg'],m['O']]]).to(torch.long).cuda()
el = ['Ca','Mg','O']

ee = torch.Tensor([1]).to(torch.long).cuda()


sampled_images = diffusion.sample(
    classes=image_classes,
    e=ee,
    iss=iss0,
    cond_scale=3.
)

data = sampled_images.to('cpu').detach().numpy()
process('./', sh, data, el)
```

