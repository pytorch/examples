"""
Definition of the model U-Net for image segmentation in X-Ray. Alseo we define a u-Net
with a smaller in_put size.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from test_tools import plot_img_tensor


class ModelUNet(nn.Module):

    '''
    This class defines the U-Net architecture, which is widely used for image segmentation tasks. 
    It includes downsampling, bottleneck, and upsampling blocks, along with a softmax activation 
    function for generating probability maps.


    Attributes
    ----------
        downsampling1 : nn.Sequential
            A sequential aplication of conv layers.
        downsampling2 : nn.Sequential
            A sequential aplication of MaxPool2d layers, and Conv layers.
        downsampling3 : nn.Sequential
            A sequential aplication of MaxPool2d layers, and Conv layers.
        downsampling4 : nn.Sequential
            A sequential aplication of MaxPool2d layers, and Conv layers.
        center_block : nn.Sequential
            A sequential aplication of MaxPool2d, Conv, and ConvTranspose2d layers.
        upsampling1 : nn.Sequential
            A sequential aplication of Conv, and ConvTranspose2d
        upsampling2 : nn.Sequential
            A sequential aplication of Conv, and ConvTranspose2d
        upsampling3 : nn.Sequential
            A sequential aplication of Conv, and ConvTranspose2d
        upsampling4 : nn.Sequential
            A sequential aplication of Conv, and ConvTranspose2d
        softMax : nn.Softmax
            A softmax activation function for get probabilitys in the end of the model


    Methods
    -------
    ConvBlock(in_channels: int, out_channels: int, kernel_size: int, stride: int) -> nn.Sequential
        Creates a convolutional block with Conv2d, SiLU, BatchNorm2d, and another Conv2d layer.
    ConvUpBlock(in_channels: int, out_channels: int, kernel_size: int, stride: int) -> nn.Sequential
        Creates an upsampling block with ConvTranspose2d, BatchNorm2d, and SiLU.
    forward(in_put: torch.Tensor) -> torch.Tensor
        Defines the forward pass of the U-Net model, returning the segmentation out_put.
    '''

    def __init__(
            self, 
            height : int = 572,
            width  : int = 572,
            in_channels: int = 1,
        ):
        super(ModelUNet, self).__init__()

        #* -> -> (U-net architecture arrows)
        self.downsampling1 = self.ConvBlock(in_channels , 64, 3, 1)  

        #* ↓ -> ->
        self.downsampling2 = nn.Sequential(
                                            nn.MaxPool2d(kernel_size = 2),
                                            self.ConvBlock(64 , 128, 3, 1)
                                        )

        #* ↓ -> ->
        self.downsampling3 = nn.Sequential(
                                            nn.MaxPool2d(kernel_size = 2),
                                            self.ConvBlock(128, 256, 3, 1)
                                        )

        #* ↓ -> ->
        self.downsampling4 = nn.Sequential(
                                            nn.MaxPool2d(kernel_size = 2),
                                            self.ConvBlock(256, 512, 3, 1)
                                        )

        #* ↓ -> ->  ↑
        #* use pixshuffle ? 
        self.center_block = nn.Sequential(
                                            nn.MaxPool2d(kernel_size = 2),
                                            self.ConvBlock(512 , 1024, 3, 1),
                                            self.ConvUpBlock(1024, 512, 3, 2)
                                        )

        #* ↑ -> -> 
        self.upsampling1 = nn.Sequential(
                                            self.ConvBlock(1024, 512, 3, 1),
                                            self.ConvUpBlock(512, 256, 3, 2)
                                        )

        #* ↑ -> -> 
        self.upsampling2 = nn.Sequential(
                                            self.ConvBlock(512, 256, 3, 1),
                                            self.ConvUpBlock(256, 128, 3, 2)
                                        )

        #* ↑ -> -> 
        self.upsampling3 = nn.Sequential(
                                            self.ConvBlock(256, 128, 3, 1),
                                            self.ConvUpBlock(128, 64, 3 ,2)
                                        )

        #* ↑ -> -> 
        self.upsampling4 = nn.Sequential(
                                            self.ConvBlock(128, 64, 3, 1), 
                                            self.ConvBlock(64, 2, 1, 1)
                                        )

        self.softMax  = nn.Softmax(dim = 1)

    def ConvBlock(
            self, 
            in_channels  : int,
            out_channels : int, 
            kernel_size  : int, 
            stride : int,
        ) -> nn.Sequential:
        '''
            This method will return an sequential with Conv2d, SiLU, BatchNorm2d, 
            Conv2d, and SilU.

            Args
            ----
                in_channels : int
                    The number of input channels.
                out_channels : int
                    The number of output channels.
                kernel_size : int
                    The size of the convolutional kernel.
                stride : int
                    The stride of the convolutional operation.

            Returns
            -------
                nn.Sequential
                    A sequential container with the defined layers.
        '''

        return nn.Sequential(
            nn.Conv2d(
                    in_channels  = in_channels,
                    out_channels = out_channels,
                    kernel_size  = kernel_size,
                    stride = stride
                    ), 
            nn.SiLU(), #* the origin model use Relu
            nn.BatchNorm2d(out_channels), #* The origin model do not use batchNorm
            nn.Conv2d(
                    in_channels  = out_channels,
                    out_channels = out_channels,
                    kernel_size  = kernel_size,
                    stride = stride
                    ), 
            nn.SiLU() #* the origin model use Relu
        )

    def ConvUpBlock(
            self, 
            in_channels  : int,
            out_channels : int, 
            kernel_size  : int, 
            stride : int,
        ) -> nn.Sequential:
        '''
            This method will return an sequential with ConvTransposed2d, BatchNorm2d,
            and SiLU.

            Args
            ----
                in_channels : int
                    The number of input channels.
                out_channels : int
                    The number of output channels.
                kernel_size : int
                    The size of the convolutional kernel.
                stride : int
                    The stride of the convolutional operation.
            
            Returns
            -------
                nn.Sequential
                    A sequential container with the defined layers.
        '''

        return nn.Sequential(
            nn.ConvTranspose2d(
                            in_channels  = in_channels, 
                            out_channels = out_channels, 
                            kernel_size  = kernel_size, 
                            stride  = stride,
                            padding = 1,
                            output_padding = 1
                            ),
            nn.BatchNorm2d(out_channels), 
            nn.SiLU()
        )

    def forward(
            self, 
            in_put : torch.Tensor,
    ) -> torch.Tensor:
        '''
            This method will compute the model out_put, and return it.

            Args
            ----
                in_put : torch.Tensor
                    A img batch tensor of shape (batch size, self.in_channels, 572, 572).
            
            Returns
            -------
                out_put : torch.Tensor
                    The model out_put of shape (batch size, 2, 572, 572).
        '''

        out_put = self.downsampling1(in_put)                  #* -> -> 
        copy1  = transforms.Resize((392, 392))(out_put)      

        out_put = self.downsampling2(out_put)                 #* ↓ -> ->
        copy2  = out_put[:, :, 40:240, 40:240]

        out_put = self.downsampling3(out_put)                 #* ↓ -> ->
        copy3  = out_put[:, :, 16:120, 16:120] 

        out_put = self.downsampling4(out_put)                 #* ↓ -> ->
        copy4  = out_put[:, :, 4:60, 4:60] #todo maxPol ? 

        out_put = self.center_block(out_put)                   #* ↓ -> ->  ↑

        out_put = torch.cat((out_put, copy4), dim=1)          #* concatenate the tensors
        out_put = self.upsampling1(out_put)                   #* ↑ -> ->  

        out_put = torch.cat((out_put, copy3), dim = 1) 
        out_put = self.upsampling2(out_put)                   #* ↑ -> ->  

        out_put = torch.cat((out_put, copy2), dim= 1)
        out_put = self.upsampling3(out_put)                   #* ↑ -> ->  

        out_put = torch.cat((out_put, copy1), dim= 1)
        out_put = self.upsampling4(out_put)                   #* ↑ -> ->  

        out_put = self.softMax(out_put)
        return out_put


#TODO make a U-Net for every in_put size

class modeluNet(ModelUNet):
    '''
        Implementation of the model U-Net, but with other in_put size. For have 
        little U-Net model.

        Methods 
            OutPutsCopys(in_put)
                This function will return the model prediction, and the out_put of the
                downsampling's layers.
    '''

    def __init__(
            self, 
            height: int = 252, 
            width : int = 252,
            in_channels: int = 1
        ) -> None:
        super().__init__(height, width, in_channels)

    def OutPutsCopys(
            self, 
            in_put: torch.Tensor,
    ):
        '''
            This function will compute and return the model out_put, and the out_puts of 
            the fist four blocks. 

            Args
            ----
                in_put : torch.Tensor
                    A img batch tensor of shape (batch size, self.in_channels, 252, 252).

            Returns
            -------
            A tuple with (out_put, copy1, copy2, copy3, copy4), where out_put is the model 
            out_put, copy1 is the out_put of the fist convBlock, ... , copy4 is the out_put 
            of the fourth convBlock.

        '''

        out_put = self.downsampling1(in_put)                   #* -> -> 
        copy1  = transforms.Resize((72, 72))(out_put)

        out_put = self.downsampling2(out_put)                  #* ↓ -> ->
        copy2  = transforms.Resize((40, 40))(out_put)

        out_put = self.downsampling3(out_put)                  #* ↓ -> ->
        copy3  = transforms.Resize((24, 24))(out_put)  

        out_put = self.downsampling4(out_put)                  #* ↓ -> ->
        copy4  = transforms.Resize((16, 16))(out_put)  

        out_put = self.center_block(out_put)                    #* ↓ -> ->  ↑

        out_put = torch.cat((out_put, copy4), dim=1)           #* concatenate the tensors
        out_put = self.upsampling1(out_put)                    #* ↑ -> ->

        out_put = torch.cat((out_put, copy3), dim = 1)  
        out_put = self.upsampling2(out_put)                    #* ↑ -> ->

        out_put = torch.cat((out_put, copy2), dim= 1)
        out_put = self.upsampling3(out_put)                    #* ↑ -> ->

        out_put = torch.cat((out_put, copy1), dim= 1)          
        out_put = self.upsampling4(out_put)                    #* ↑ -> ->

        out_put = self.softMax(out_put)                        #* get probabilitys
        return out_put, copy1, copy2, copy3, copy4

    def forward(
            self, 
            in_put: torch.Tensor,
    ):
        '''
            This function will compute and return the model out_put.

            Args
            ----
                in_put : torch.Tensor
                    A img batch tensor of shape (batch size, self.in_channels, 252, 252).
            
            Returns
            -------
                out_put : torch.Tensor
                    The model out_put of shape (batch size, 2, 252, 252).
        '''

        out_put = self.downsampling1(in_put)                   #* -> -> 
        copy1  = transforms.Resize((72, 72))(out_put)

        out_put = self.downsampling2(out_put)                  #* ↓ -> ->
        copy2  = transforms.Resize((40, 40))(out_put)

        out_put = self.downsampling3(out_put)                  #* ↓ -> ->
        copy3  = transforms.Resize((24, 24))(out_put)  

        out_put = self.downsampling4(out_put)                  #* ↓ -> ->
        copy4  = transforms.Resize((16, 16))(out_put)  

        out_put = self.center_block(out_put)                    #* ↓ -> ->  ↑

        out_put = torch.cat((out_put, copy4), dim=1)           #* concatenate the tensors
        out_put = self.upsampling1(out_put)                    #* ↑ -> ->

        out_put = torch.cat((out_put, copy3), dim = 1)  
        out_put = self.upsampling2(out_put)                    #* ↑ -> ->

        out_put = torch.cat((out_put, copy2), dim= 1)
        out_put = self.upsampling3(out_put)                    #* ↑ -> ->

        out_put = torch.cat((out_put, copy1), dim= 1)          
        out_put = self.upsampling4(out_put)                    #* ↑ -> ->

        out_put = self.softMax(out_put)                        #* get probabilitys
        return out_put