import torch
from torchvision.transforms.functional import center_crop
from torch.nn import Module, ReLU, Sequential, Conv2d, MaxPool2d, ConvTranspose2d


class Unet (Module): 
    def __init__(self, input_channels: int, output_channels : int) -> None : 
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels


        self.block_1_down = Sequential(
            Conv2d(in_channels= self.input_channels, out_channels= 64, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(in_channels= 64, out_channels= 64, kernel_size=3, padding=1),
            ReLU()
        )

        self.block_2_down = Sequential(
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(in_channels= 64, out_channels= 128, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(in_channels= 128, out_channels= 128, kernel_size=3, padding=1),
            ReLU()
        )

        self.block_3_down = Sequential(
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(in_channels= 128, out_channels= 256, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(in_channels= 256, out_channels= 256, kernel_size=3, padding=1),
            ReLU()
        )

        self.block_4_down = Sequential(
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(in_channels= 256, out_channels= 512, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(in_channels= 512, out_channels= 512, kernel_size=3, padding=1),
            ReLU()
        )

        self.block_5 = Sequential(
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(in_channels= 512, out_channels= 1024, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(in_channels= 1024, out_channels= 1024, kernel_size=3, padding=1),
            ReLU(),
            ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        )

        self.block_4_up = Sequential(
            Conv2d(in_channels= 1024, out_channels= 512, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(in_channels= 512, out_channels= 512, kernel_size=3, padding=1),
            ReLU(),
            ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        )
        
        self.block_3_up = Sequential(
            Conv2d(in_channels= 512, out_channels= 256, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(in_channels= 256, out_channels= 256, kernel_size=3, padding=1),
            ReLU(),
            ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        )

        self.block_2_up = Sequential(
            Conv2d(in_channels= 256, out_channels= 128, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(in_channels= 128, out_channels= 128, kernel_size=3, padding=1),
            ReLU(),
            ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        )

        self.block_1_up = Sequential(
            Conv2d(in_channels= 128, out_channels= 64, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(in_channels= 64, out_channels= 64, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(in_channels= 64, out_channels= self.output_channels, kernel_size=1),
            # Softmax(dim=0)
        )

    def forward (self, img : torch.Tensor) -> torch.Tensor :
        assert(img.shape[0] == self.input_channels), "input channel doesnt match"

        b1_out = self.block_1_down(img)
        # print(f"b1 : {b1_out.shape}")

        b2_out = self.block_2_down(b1_out)
        # print(f"b2 : {b2_out.shape}")
        
        b3_out = self.block_3_down(b2_out)
        # print(f"b3 : {b3_out.shape}")

        b4_out = self.block_4_down(b3_out)
        # print(f"b4 : {b4_out.shape}")

        x = self.block_5(b4_out)
        # print(f"b5 : {x.shape}")

        # crop & concatenate with b4_out 
        # b4_out_cropped = center_crop(b4_out, x.shape[1:])
        x = torch.concat([b4_out, x], dim=0)
        x = self.block_4_up(x)
        # print(f"b4 : {x.shape}")

        # crop & concatenate with b3_out
        # b3_out_cropped = center_crop(b3_out, x.shape[1:])
        x = torch.concat([b3_out,x], dim=0)
        x = self.block_3_up(x)
        # print(f"b3 : {x.shape}")

        # crop & concatenate with b2_out
        # b2_out_cropped = center_crop(b2_out, x.shape[1:])
        x = torch.concat([b2_out,x], dim=0)
        x = self.block_2_up(x)
        # print(f"b2 : {x.shape}")

        # crop & concatenate with b1_out
        # b1_out_cropped = center_crop(b1_out, x.shape[1:])
        x = torch.concat([b1_out,x], dim=0)
        x = self.block_1_up(x)
        # print(f"b1 : {x.shape}")
        
        return x


if __name__ == "__main__": 
    in_channels = 3
    out_channels = 2

    height = 512
    width = 512

    model = Unet(in_channels, out_channels)

    # parameters = model.parameters()
    # for  para in parameters : 
    #     print(para)
    #     # print(f"Name : {name}\tParameters : {para}")

    output = model(torch.rand(in_channels, height, width))

    print(output.shape)