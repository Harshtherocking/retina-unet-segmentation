import torch 
import os
from torch.optim import AdamW
from torch.utils.data import DataLoader

from utils import * 
from dataloader import ImageDataset, PATH
from torchvision.utils import save_image

from tqdm import tqdm

ModelDir = os.path.join(os.getcwd(), "models") 
CheckpointsDir = os.path.join(os.getcwd(), "checkpoints")
ResultDir = os.path.join(PATH, "results")

ds= ImageDataset(path= os.path.join(PATH, "train"))
dl= DataLoader(ds, batch_size=1, num_workers=4, pin_memory= True)

