"""
    This file contains the class DataSetImgToImg, which is a PyTorch Dataset for loading images and their corresponding masks.
    The class is designed to work with image-to-image translation tasks, such as segmentation or style transfer.
"""
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


identity_transform = transforms.Compose([
                        #* ToPILImage = cahnge the data type from PyTorch tensor or a NumPy ndarray to : A PIL (Python Imaging Library)
                        transforms.ToPILImage(),
                        #* change the data type from Numpy or PIL to tensor
                        transforms.ToTensor()
                    ])


class DataSetImgToImg(Dataset):
    '''
        A dataset of models from images to img. We need the path of the img 
        input of the model and img outPut i.e path of img folder and mask folder.

        Attributes
        ----------
            data : list[tuple]
                The list of tuples with img name, and mask. Example 
                [(in_put_img.1.jpg, out_put_img1.1.jpg)].
            root_data : list[str, str]
                The root_data[0] is a path to the data images folder for the inPut.
                The root_data[1] is a path to the data images folder for the outPut.
            trans_for_in_put_img  : torchvision.transforms.Compose, optional
                Transformation for the inPut images
            trans_for_out_put_img : torchvision.transforms.Compose, optional
                Transformation for the outPut images
            test : bool
                If is true we will return a dataset of 'data_size' elements for do testing
        Methods
        -------
            __getitem__(index):
                Fetches and transforms the input and output images at the specified index.
    
            __len__(void) -> int:
                Return size of the data set.
    '''

    def __init__(
            self, 
            root_data,
            trans_for_in_img  = identity_transform, 
            trans_for_out_img = identity_transform, 
            test     = False, 
            data_size = 100
    ):
        super(DataSetImgToImg, self).__init__()

        self.data = []
        self.root_data   = root_data
        self.trans_for_in_put_img  = trans_for_in_img
        self.trans_for_out_put_img = trans_for_out_img
        self.test = test
        
        #* Create a list of the name of the files in the root_datas.
        #TODO if the names are diferents this do now work well
        in_put_images  = os.listdir(self.root_data[0])
        out_put_images = os.listdir(self.root_data[1])

        if(len(in_put_images) != len(out_put_images)):
            print("len(in_put_images) != len(out_put_images)")

        if(self.test == True):
            data_size = min(len(in_put_images), data_size)
            in_put_images  =  in_put_images[0:data_size]
            out_put_images = out_put_images[0:data_size]

        #* Save a list of tuples like [(in_put_img.1.jpg, out_put_img1.1.jpg)]
        self.data = list(zip(in_put_images, out_put_images))
        print("Size data set lower definition", len(in_put_images))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        #* Read the images, transform them into an array, and only use the first 3 channels.
        in_put_img_file = self.data[index][0]
        in_put_img_pth = os.path.join(self.root_data[0] + '/' + in_put_img_file)
        in_put_img  = np.array(Image.open(in_put_img_pth))
        in_put_img  = in_put_img [:, :, :3]
        
        out_put_img_file = self.data[index][1]
        out_put_img_pth = os.path.join(self.root_data[1] + '/' + out_put_img_file)
        out_put_img  = np.array(Image.open(out_put_img_pth))
        # chanels = min(np.array(Image.open(out_put_img_pth)).shape[0], 1)

        out_put_img  = out_put_img [:, :]

        #* Apply the corresponding trasformations. Could be data aumentation functions
        in_put_img  = self.trans_for_in_put_img(in_put_img)
        out_put_img = self.trans_for_out_put_img(out_put_img)


        return in_put_img, out_put_img

