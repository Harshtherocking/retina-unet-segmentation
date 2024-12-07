import torch 
import os
from torch.optim import AdamW

from utils import * 
from dataloader import ImageDataloader, PATH
from unet import Unet

from tqdm import tqdm


ModelDir = os.path.join(os.getcwd(), "models")
CheckpointsDir = os.path.join(os.getcwd(), "checkpoints")

train_path = os.path.join(PATH , "ImageSets", "480p", "train.txt")



def train(
        train_path: str, 
        num_epochs: int, 
        optimizer: torch.optim.Optimizer, 
        model: torch.nn.Module, 
        model_dir: str, 
        checkpoint_dir: str
        ) -> torch.nn.Module:
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "device")
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss() 


    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    dataloader = ImageDataloader(path= train_path)

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}... Training in progress.")

        for i in tqdm(range(len(dataloader))): 
            x,y = dataloader[i]

            x = x.to(device)
            y = y.to(device)

            total_pixels = y.numel()
            
            pred = model(x)
            # print(f"{i}\tPred : \n{pred}")

            optimizer.zero_grad()
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            
            print(f"{i + 1}\tLoss : {loss}\nAverage Loss per pixel : {loss/total_pixels}") if not i % 100 or i == len(dataloader) - 1 else None


        model_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        save_checkpoint(model_state, checkpoint_dir, f"checkpoint_epoch_{epoch}.pkl")


        # Save the latest model after each epoch
        torch.save(model.state_dict(), os.path.join(model_dir, "model_last.pkl"))

        # Optionally save the best model based on validation performance (not implemented here)
        # Example: if val_loss < best_loss: save_checkpoint(...)

    print("Training completed.")
    return model



if __name__ == "__main__" :
    unet = Unet(3,2)
    adam = AdamW(unet.parameters(), lr=0.01, amsgrad= True)
    epochs = 5

    train (train_path, epochs, adam, unet, ModelDir, CheckpointsDir)
