# CrystalDiffusion

The maintenance of this project is handled within the following repository:
https://github.com/PhysiLearn/CrystalDiffusion

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
## Citation

    @misc{denoising-diffusion-pytorch ,
        author  = {lucidrains},
        url     = {https://github.com/lucidrains/denoising-diffusion-pytorch}
    }

    @ARTICLE{2024arXiv240113192L,
           author = {{Li}, Zhelin and {Mrad}, Rami and {Jiao}, Runxian and {Huang}, Guan and {Shan}, Jun and {Chu}, Shibing and {Chen}, Yuanping},
            title = "{Generative Design of Crystal Structures by Point Cloud Representations and Diffusion Model}",
          journal = {arXiv e-prints},
         keywords = {Computer Science - Artificial Intelligence, Condensed Matter - Materials Science, Computer Science - Machine Learning, Physics - Computational Physics},
             year = 2024,
            month = jan,
              eid = {arXiv:2401.13192},
            pages = {arXiv:2401.13192},
              doi = {10.48550/arXiv.2401.13192},
    archivePrefix = {arXiv},
           eprint = {2401.13192},
     primaryClass = {cs.AI},
           adsurl = {https://ui.adsabs.harvard.edu/abs/2024arXiv240113192L},
          adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }
