import torch 
import os
from torch.optim import AdamW

from utils import * 
from dataloader import ImageDataset, PATH
from unet import Unet

from torchvision.io.image import write_png

from torch.nn.functional import softmax
from torch.utils.data import DataLoader

from tqdm import tqdm

model_dir = os.path.join(os.getcwd(), "models")
checkpoint_dir= os.path.join(os.getcwd(), "checkpoints")
result_dir = os.path.join(os.getcwd(), "results")

os.makedirs(model_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
# os.makedirs(result_dir, exist_ok=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "device")
model = Unet(3,2).to(device)

optimizer = AdamW(model.parameters(), lr=0.001, amsgrad=True)

loss_fn = torch.nn.CrossEntropyLoss()

ds= ImageDataset(path= os.path.join(PATH, "train"))

dl = DataLoader(ds, batch_size=2, pin_memory= True, num_workers= 4)

epochs = 1000

torch.cuda.empty_cache()

if __name__ == "__main__" :

    for epoch in tqdm(range(1, epochs + 1)):
        print(f"Epoch {epoch}/{epochs}")

        for x,y  in dl : 
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            
            optimizer.zero_grad()
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()

            # if epoch % 10 == 0: 
            #     pred_softmax = softmax(pred,1).cpu()
            #     image_pred_softmax = (pred_softmax[0,1,:,:].unsqueeze(0) * 255).to(dtype=torch.uint8)

            #     write_png(image_pred_softmax, os.path.join(result_dir, f"epoch_{epoch}.png"))  
                
            if epoch % 10 == 0 : 
                print(f"Loss : {loss}")

            torch.cuda.empty_cache()


        if epoch % 100 == 0 : 
            model_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }
            save_checkpoint(model_state, checkpoint_dir, f"checkpoint_epoch_{epoch}.pkl")

        torch.save(model.state_dict(), os.path.join(model_dir, "last_model.pkl"))

        torch.cuda.empty_cache()


    print("Training completed.")
