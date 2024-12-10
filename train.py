import torch 
import os
from torch.optim import AdamW

from utils import * 
from dataloader import ImageDataloader, PATH
from unet import Unet

from torchvision.io.image import write_png

from torch.nn.functional import softmax

from tqdm import tqdm
import random

random.seed(42)

model_dir = os.path.join(os.getcwd(), "models")
checkpoint_dir= os.path.join(os.getcwd(), "checkpoints")
result_dir = os.path.join(os.getcwd(), "results")

os.makedirs(model_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "device")
model = Unet(3,2).to(device)

optimizer = AdamW(model.parameters(), lr=0.01, amsgrad=True)

loss_fn = torch.nn.CrossEntropyLoss()

dataloader = ImageDataloader(path= PATH)

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5)

epochs = 1000

torch.cuda.empty_cache()

if __name__ == "__main__" :

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}... Training in progress.")

        # for i in tqdm(range(len(dataloader))): 
        for i in range(0,1) : 
            x,y = dataloader[56]

            x = x.to(device)
            y = y.to(device)


            pred = model(x)

            optimizer.zero_grad()
            loss = loss_fn(pred, y)

            loss.backward()

            pred_softmax = softmax(pred,0).cpu()
            image_pred_softmax = (pred_softmax[1,:,:].unsqueeze(0) * 255).to(dtype=torch.uint8)

            
            write_png(image_pred_softmax, os.path.join(result_dir, f"epoch_{epoch}.png"))  


            # with torch.no_grad():
            #     for name, param in model.named_parameters():
            #         print(f"{name} Gradient Norm: {param.grad.norm()}")
            #         print(f"Before Step - {name}: {param}")
                
            #     optimizer.step()

            #     for name, param in model.named_parameters():
            #         print(f"After Step - {name}: {param}")

            optimizer.step()
            # scheduler.step(loss)
            # print(f"last lr : {scheduler.get_last_lr()}")
            
            print(f"{i + 1}\tLoss : {loss}") if not i % 100 or i == len(dataloader) - 1 else None

            # for i, para in enumerate(model.parameters()) : 
            #     print(f"{i} : {id(para)}")

            # print(model.parameters())
            # for param in model.parameters() : 
                # print(param.grad)

            torch.cuda.empty_cache()



        # model_state = {
        #     'epoch': epoch,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'loss': loss,
        # }

        # save_checkpoint(model_state, checkpoint_dir, f"checkpoint_epoch_{epoch}.pkl")

        # torch.save(model.state_dict(), os.path.join(model_dir, "model_last.pkl"))

        # # save image for each epoch 
        # x, y = dataloader[random.randint(0,len(dataloader))]
        # x = x.to(device)
        # pred = model(x).cpu()
        
        torch.cuda.empty_cache()


    print("Training completed.")
