import pandas as pd
import os

import torch 
from torchvision.io import read_image, write_png
from torch.utils.data import Dataset
from utils import *
 

from torchvision.transforms.functional import to_pil_image
from torchvision.utils import save_image
from PIL import Image


PATH = os.path.join(os.getcwd() , "Retina", "train")

class ImageDataloader (Dataset): 
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

        # print(f"Image : {image.shape}")
        # print(f"Mask: {mask.shape}")

        # if mask.shape[0] == 2 : 
        #     mask = mask[0,:,:]
        #     mask = torch.reshape(mask, (1,*mask.shape))
        
        # pil_image = to_pil_image(image)
        # pil_mask = to_pil_image(mask)

        # pil_image.show()
        # pil_mask.show()

        return uint8_to_float32(image)/255, image_to_binary_mask(mask)


if __name__ == "__main__": 

    loader = ImageDataloader(PATH)
    l = len(loader)
    x,y = loader[l-1]

    print(y.dtype)
    print(y.shape)

    y = (y[1,:,:].unsqueeze(0) *255).to(dtype=torch.uint8).cpu()
    write_png(y,"sample.png")
    print(y.shape)

