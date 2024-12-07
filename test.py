import torch 
import os
from torch.optim import AdamW

from utils import * 
from dataloader import ImageDataloader, PATH
from unet import Unet
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image

from PIL import Image

from tqdm import tqdm

ModelDir = os.path.join(os.getcwd(), "models")
CheckpointsDir = os.path.join(os.getcwd(), "checkpoints")

test_path = os.path.join(PATH , "ImageSets", "480p", "val.txt")

ResultDir = os.path.join(PATH, "results")

def test (
        test_path : str, 
        model : torch.nn.Module,
        result_dir : str
        ) -> None :
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "device")
    model.to(device)

    os.makedirs(result_dir, exist_ok= True)

    dataloader = ImageDataloader(path = test_path)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = 0

    for i in tqdm(range(len(dataloader))) :     
        if i != 340 : 
            continue
        x, y = dataloader[i]

        x = x.to(device)
        y = y.to(device)

        out = model(x)
        print(f"Loss : {loss_fn(out, y)}")
        print(f"output : \n{out}\n{out.shape}")

        out = softmax_to_binary_masks(out)
        print(f"Binary output : \n{out}\n{out.shape}")
        
        out = binary_mask_to_image(out)
        print(f"Image output : \n{out}\n{out.shape}")

        print("max element : ", out.max())
        
        out.to("cpu")

        img = to_pil_image(out)
        img.show()

        # save_image(out, os.path.join( result_dir, f"{i}.jpeg"))

        return

        loss += loss_fn(out, y).item()
    
    print(f"Average Loss : {loss/len(dataloader)}")


if __name__ == "__main__" : 
    unet = Unet(3,2)
    checkpt = load_checkpoint(CheckpointsDir, "checkpoint_epoch_5.pkl")
    # states = torch.load(os.path.join(ModelDir, "model_last.pkl"),weights_only=True)
    states = checkpt["model_state_dict"]

    unet.load_state_dict(states)
    
    test(test_path, unet, ResultDir)


