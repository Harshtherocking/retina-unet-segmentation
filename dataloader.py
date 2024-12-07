import pandas as pd
import os

import torch 
from torchvision.io import read_image 
from torch.utils.data import Dataset
from utils import *


PATH = os.path.join(os.getcwd() , "DAVIS-data", "DAVIS")

class ImageDataloader (Dataset): 
    def __init__(self, path : str) -> None:
        data = pd.read_csv(path, sep = " ", header= None)
        self.images = data.iloc[:,0].to_list()
        self.annotations = data.iloc[:,1].to_list()
        self.len = len(self.images)
    
    def __len__(self) -> int :
        return self.len

    def __getitem__(self, index : int ) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = os.path.join(PATH, self.images[index][1:])
        anno_path = os.path.join(PATH, self.annotations[index][1:])

        image = read_image(img_path)
        annotation = read_image(anno_path)

        image = pad_right_bottom(image)
        annotation = pad_right_bottom(annotation)

        if annotation.shape[0] == 2 : 
            annotation = binary_mask_to_image(annotation)
        return uint8_to_float32(image), image_to_binary_mask(annotation)



if __name__ == "__main__": 
    train_path = os.path.join(PATH , "ImageSets", "480p", "train.txt")
    loader = ImageDataloader(train_path)

    x , y = loader[77]

    print(x.shape)
    print(y.shape)

