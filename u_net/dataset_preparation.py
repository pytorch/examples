'''	
    This script is used to prepare the Brain MRI segmentation dataset for training a model 
    image to image. It creates the following folders within 'new_folders_path': train/img, 
    train/mask, validation/img, and validation/mask. After that the function copies the 
    original images into train/img and train/mask. When copying the mask imgaes, "_mask" 
    is removed from their filenames to match with the images filesnames (this is necesary 
    for  use tiwh the DataSetImgToImg class). Subsequently, the data is split into 80% for 
    training and 20% for validation. The function 'preparation_brain_MRI_set' is the 
    main function that takes the path of the dataset and the new folder path as input arguments.    
    It creates the necessary folders and copies the images into them. The function 'split_data_set' 
    is used to split the dataset into training and validation sets. The function 'copy_in_new_file' 
    is used to copy the images from one folder to another and remove the old file if specified.
'''

import os
import random


def copy_in_new_file(
        old_path : str, 
        new_path : str, 
        img_name : str, 
        delete_old_file : bool = False,
) -> None:
    '''
        This function copy the img from  direction old_path/img_name in to new_path/img_name2
        where img_name2 = img_name with out "_mask" string. If delete_old_file = true the old
        img will be remove. 

        Args
        ----
            old_path : str
                The folder where is the img.
            new_path : str
                The folder where we will do the copy of the img.
            img_name : str
                The img name.
            delete_old_file : bool = False
                delete_old_file = true if we what to delete the old file
    '''

    with open(old_path + '/' + img_name, "rb") as f:
        img_copy = f.read()

    if("mask" in img_name):
        img_name = img_name.replace("_mask", "")

    with open(new_path + '/' + img_name, "wb") as f:
        
        f.write(img_copy)

    if delete_old_file == True:
        #TODO test
        os.remove(old_path)


def split_data_set(new_folders_path : str) -> None:
    '''
        This function that split the data set in to train and validation folders. 
        The validation set will be the 20% of the origen data set. We need train
        and validation folders in new_folders_path and img, mask folders in train
        and validation respectivy. And the 100 fo the data set in train. Like this:

        new_folders_path/
        |
        |--train/
        |    |
        |    |- img      
        |    |
        |    L mask      
        |
        |--validation
        |    |
        |    |- img      
        |    |
        |    L mask

        Args:
        -----
            new_folders_path : str
                The folder where is the data set with the folders train with img folder and mask folder
                and the folder validation with the folders img and mask (like the diagram).
    '''

    files_img  = os.listdir(new_folders_path + "/train/img")
    files_mask = os.listdir(new_folders_path + "/train/mask")
    data_set_size = len(files_img)     #* number of pairs (img, mask) in the data set
    validation_size = data_set_size//5 #todo add the arg porcentage.

    for _ in range(validation_size):

        img_name = random.choice(files_img)
        old_path = new_folders_path + "/train/img" 
        new_path = new_folders_path + "/validation/img"
        copy_in_new_file(old_path = old_path, new_path = new_path, img_name = img_name)
        files_img.remove(img_name)

        old_path = new_folders_path + "/train/mask" 
        new_path = new_folders_path + "/validation/mask"
        copy_in_new_file(old_path = old_path, new_path = new_path, img_name = img_name)
        files_mask.remove(img_name)
    
    print("validation_size = ", validation_size)
    print("trainSize = "     , len(files_img))


def preparation_brain_MRI_set(
        path_data_set : str, 
        new_folders_path : str,
) -> None:
    '''
        This function prepare the Brain MRI segmentation dataset in the specified
        'new_folders_path' for training a model image to image. It creates the 
        following folders within 'new_folders_path': train/img, train/mask, 
        validation/img, and validation/mask. After that the function copies the
        original images into train/img and train/mask. When copying the mask 
        imgaes, "_mask" is removed from their filenames to match with the images 
        filesnames (this is necesary for  use tiwh the DataSetImgToImg class).
        Subsequently, the data is split into 80% for training and 20% for validation.

        Args:
        -----
            path_data_set : str
                The foder of the data set Brain MRI 
            new_folders_path : str
                    The folder where we will create the folders train/img, train/mask \n
                    validation/img, and validation/mask. For afther move the data set \n
                    in the train folder, and after split the data set in train, and
                    validation folders.
    '''

    folder_kaggle_3m = os.listdir(path_data_set)
    path_data_set += '/' + folder_kaggle_3m[0]
    folders_list  = os.listdir(path_data_set)[2:] #* ignore the red and csv

    os.makedirs(new_folders_path + "/train/img" , exist_ok=True) #*Create the folders for save the imgs
    os.makedirs(new_folders_path + "/train/mask", exist_ok=True) #*Create the folders for save the imgs
    os.makedirs(new_folders_path + "/validation/img" , exist_ok=True) #*Create the folders for save the imgs
    os.makedirs(new_folders_path + "/validation/mask", exist_ok=True) #*Create the folders for save the imgs

    data_set_size = 0
    #* Open all the img in the dataSet and separate in img and mask. 
    for folder in (folders_list):
        img_folder_list = os.listdir(path_data_set + '/' + folder)
        for  img_name in (img_folder_list):
            data_set_size += 1
            if("mask" in img_name):
                copy_in_new_file(path_data_set + '/' +  folder, new_folders_path + "/train/mask", img_name)
            else:
                copy_in_new_file(path_data_set + '/' +  folder, new_folders_path + "/train/img", img_name)

    split_data_set(new_folders_path)


