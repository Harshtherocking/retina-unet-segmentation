import pandas as pd
import os

import torch 
from torchvision.io import read_image, write_png
from torch.utils.data import Dataset
from utils import *
 

PATH = os.path.join(os.getcwd() , "Retina")

class ImageDataset (Dataset): 
    def __init__(self, path : str) -> None:

        self.img_path = os.path.join(path, "image")
        self.mask_path = os.path.join(path, "mask")

        self.img_list = os.listdir(self.img_path)
        self.mask_list = os.listdir(self.mask_path)

        self.len = len(self.img_list)
    
    def __len__(self) -> int :
        return self.len

    def __getitem__(self, index : int ) -> tuple[torch.Tensor, torch.Tensor]:

        image = read_image(os.path.join(self.img_path, self.img_list[index]))
        mask = read_image(os.path.join(self.mask_path, self.mask_list[index]))

        return uint8_to_float32(image), image_to_binary_mask(mask).to(dtype=torch.long)


if __name__ == "__main__": 

    path = os.path.join(PATH, "train")
    loader = ImageDataset(path)
    l = len(loader)
    x,y = loader[0]

    print(x.shape)
    print(y)
    print(y.max())
    print(y.shape)

    write_png((y.unsqueeze(0) * 255).to(torch.uint8), "example.png")
