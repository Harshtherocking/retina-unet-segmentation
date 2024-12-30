import torch
from torchvision.transforms.functional import center_crop
from torch.nn import Module, ReLU, Sequential, Conv2d, MaxPool2d, ConvTranspose2d
import torch.nn.functional as F 

class BlockDown (Module) :
    def __init__ (self, input_channels : int, output_channels : int, is_first_layer : bool = False) -> None : 
        super().__init__()
        self.is_first_layer = is_first_layer

        self.maxpool = MaxPool2d(kernel_size= 2, stride= 2)
        self.conv1 = Conv2d(in_channels= input_channels, out_channels= output_channels, kernel_size= 3, padding=1) 
        self.conv2 = Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, padding=1)
        return 
    
    def forward(self, x : torch.tensor) ->torch.tensor :  
        if not self.is_first_layer : 
            x = self.maxpool(x)

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        return F.relu(x)
        


class BlockUp(Module) :
    def __init__ (self, input_channels : int, output_channels : int,  is_last_layer : bool = False) -> None : 
        super().__init__()
        self.is_last_layer = is_last_layer
        if self.is_last_layer : 
            out = output_channels
            output_channels = input_channels

        self.conv1 = Conv2d(in_channels= 2 *input_channels, out_channels= 2*output_channels, kernel_size= 3, padding=1) 
        self.conv2 = Conv2d(in_channels= 2 *output_channels, out_channels= 2*output_channels, kernel_size=3, padding=1)

        if not is_last_layer : 
            self.convT = ConvTranspose2d(in_channels= 2*output_channels, out_channels=output_channels, kernel_size=2, stride=2)
        else : 
            self.conv3 = Conv2d(in_channels=2*output_channels, out_channels=out, kernel_size=1, stride=1)
        return 

        
    def forward(self, x : torch.tensor) ->torch.tensor :  
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x  = F.relu(x)
        return self.conv3(x) if self.is_last_layer else self.convT(x)



class Unet (Module): 
    def __init__(self, input_channels: int, output_channels : int) -> None : 
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels

        self.block_1_down = BlockDown(input_channels=self.input_channels, output_channels= 64, is_first_layer= True)

        self.block_2_down = BlockDown(input_channels=64, output_channels= 128)

        self.block_3_down = BlockDown(input_channels=128, output_channels=256)

        self.block_4_down = BlockDown(input_channels=256, output_channels=512)

        # self.block_5_down = BlockDown(input_channels=512, output_channels=1024)

        # self.block= Sequential(
        #     MaxPool2d(kernel_size=2, stride=2),
        #     Conv2d(in_channels= 1024, out_channels= 2048, kernel_size=3, padding=1),
        #     ReLU(),
        #     Conv2d(in_channels= 2048, out_channels= 2048, kernel_size=3, padding=1),
        #     ReLU(),
        #     ConvTranspose2d(in_channels=2048, out_channels=1024, kernel_size=2, stride=2)
        # )

        self.block= Sequential(
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(in_channels=  512, out_channels= 1024, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(in_channels= 1024, out_channels= 1024, kernel_size=3, padding=1),
            ReLU(),
            ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        )

        # self.block_5_up = BlockUp(input_channels= 1024, output_channels=512)

        self.block_4_up = BlockUp(input_channels=512, output_channels=256)        

        self.block_3_up = BlockUp(input_channels=256, output_channels=128)        

        self.block_2_up = BlockUp(input_channels=128, output_channels=64)        

        self.block_1_up = BlockUp(input_channels=64, output_channels=self.output_channels, is_last_layer= True)

    def forward (self, img : torch.Tensor) -> torch.Tensor :
        assert(img.shape[-3] == self.input_channels), "input channel doesnt match"

        b1_out = self.block_1_down(img)
        # print(f"b1 : {b1_out.shape}")

        b2_out = self.block_2_down(b1_out)
        # print(f"b2 : {b2_out.shape}")
        
        b3_out = self.block_3_down(b2_out)
        # print(f"b3 : {b3_out.shape}")

        b4_out = self.block_4_down(b3_out)
        # print(f"b4 : {b4_out.shape}")

        # b5_out = self.block_5_down(b4_out)
        # print(f"b5 : {b5_out.shape}")

        # x = self.block(b5_out)
        # print(f"b6 : {x.shape}")

        x = self.block(b4_out)

        # x = torch.concat([b5_out, x], dim=1)
        # x = self.block_5_up(x)
        # print(f"b5 : {x.shape}")

        x = torch.concat([b4_out, x], dim=1)
        x = self.block_4_up(x)
        # print(f"b4 : {x.shape}")

        x = torch.concat([b3_out,x], dim=1)
        x = self.block_3_up(x)
        # print(f"b3 : {x.shape}")

        x = torch.concat([b2_out,x], dim=1)
        x = self.block_2_up(x)
        # print(f"b2 : {x.shape}")

        x = torch.concat([b1_out,x], dim=1)
        x = self.block_1_up(x)
        # print(f"b1 : {x.shape}")
        
        return x


if __name__ == "__main__": 
    in_channels = 3
    out_channels = 2
    batch_size = 2

    height = 512
    width = 512

    model = Unet(in_channels, out_channels).cuda()

    output = model(torch.rand(batch_size,in_channels, height, width).cuda())
    
    print(output.shape)