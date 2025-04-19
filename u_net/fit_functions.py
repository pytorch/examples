
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader


class fitertImgToImg():
    '''
        This class is for train a img to img models and test it. 
        It saves the models at the end of each epoch and save the training history.

        .. warning::
            For use the TrainModel method the model output, and the outPut imgs of the
            data_loader need to be compatibles using the funcion criterion 
            in the TrainModel method.

        Attributes
        ----
            model : nn.Module
                The model to train.
            data_set : Data_set
                The data_set containing the training data.
            history : dict
                A dictionary with keys "train_MAE" and "val_MAE" and the values 
                are historial lists of that value. 
            device  : str
                The environment devicedevice where we will do our calculations.
            batch_size : int, optional
                The batch size used for training (default is 64)
            data_set_val : Data_set, optional
                The validation data set. 
            model_save_dir : str, optional
                The path were we save the model
            training_epochs : int
                Number of training epochs the model has undergone.
            data_loader : torch.utils.data.DataLoader
                A DataLoader make with the data_set.
            data_loader_val : torch.utils.data.DataLoader
                A DataLoader make with the data_set_val.

        Methods
        -------
            GetMAE(
                    data_loader : DataLoader, 
                    criterion : torch.nn.Module = torch.nn.CrossEntropyLoss()
                    ) -> float:
                Return the MAE in the 'DataLoader' using 'criterios'.

            TrainModel(
                        opt_model  : torch.optim.Optimizer,  
                        criterion  : torch.nn.Module = torch.nn.CrossEntropyLoss(),
                        num_epochs : int = 1
                        ) -> None:
                Train the model using 'num_epochs', 'criterion', 'opt_model', and 
                the 'data_loader' attributes class.
            plotHistory(
                        interval_train : list[int] = None, 
                        interval_validation : list[int] = None
                        )->None:
                Plot the attribute history.
            Predict(
                    index : int = 0, 
                    img_path : str = None
                    ) -> tensor:
                Compute the Prediction using the model.          
    '''

    def __init__(self,
                model : nn.Module, 
                data_set, 
                device     : str = "cpu",
                batch_size : int = 64,
                data_set_val = None, 
                model_save_dir : str =  None):
        '''
            Initializes a new instance of the class fiterImgToImg.

            Args:
                model : nn.Module
                    The model to train.
                data_set : Data_set
                    The data_set containing the training data.
                    The input img and the ouPut img need to be compatible with
                    the model input and the model_out_put respective.
                device : str
                    The environment devicedevice where we will do our calculations.
                batch_size : int, optional
                    The batch size used for training (default is 64)
                data_set_val : Data_set, optional
                    The validation data set. 
                model_save_dir : str, optional
                    The path were we save the model
        '''

        if(model_save_dir is None):
            raise ValueError('model_save_dir could not be None')

        #* Initialize history
        self.history = {
            "train_MAE" : [], #* int list
            "val_MAE"   : [], #* pair(int, int) list
            #todo add the PSNR, or SSIM, FID
        }

        #* Start the class attributes
        self.model       = model
        self.data_set     = data_set
        self.device      = device
        self.batch_size  = batch_size
        self.data_set_val = data_set_val
        self.model_save_dir  = model_save_dir
        self.training_epochs = 0

        #* Initialize data loaders
        self.data_loader = DataLoader(
                                self.data_set, 
                                batch_size  = self.batch_size,
                                num_workers = 0,
                                shuffle = True,
            )

        if(self.data_set_val is not None):
            self.data_loader_val = DataLoader(
                                        self.data_set_val,
                                        batch_size  = self.batch_size,
                                        num_workers = 0,
                                        shuffle = True,
            )
        else:
            self.data_loader_val = None

    def GetMAE(self,
            data_loader : DataLoader, 
            criterion : torch.nn.Module = torch.nn.CrossEntropyLoss(),
            ):
        '''
            Compute and return the MAE in 'data_loader' using 'criterion'.
            
            Args:
            -----
                data_loader : torch.utils.data.DataLoader  
                    index list of the batch tensors of the data_set
                criterion :  torch.nn.Module, optional
                    loss function of the model

            Returns
            -------
                Return a the value of MAE in 'data_loader' using 'criterion'.
        '''

        size_data_loader = len(data_loader.dataset)
        model_MAE = 0

        if size_data_loader == 0:
            raise ValueError('The data set should not be empty.')

        with torch.no_grad():
            for (img_input, img_out_put) in data_loader:
                img_input    =  img_input.to(self.device)
                img_out_put   = img_out_put.to(self.device)
                model_out_put = self.model(img_input)

                loss = criterion(img_out_put, model_out_put)
                model_MAE += loss.item()*self.batch_size

        return model_MAE/size_data_loader

    def TrainModel(
                self,
                opt_model  : torch.optim.Optimizer, 
                criterion  : torch.nn.Module = torch.nn.CrossEntropyLoss(),
                num_epochs : int  = 1,
                get_val_MAE  : bool = False
    ):
        '''
            Train the model in the device using 'num_epochs', 'criterion', 
            'opt_model' and the dataloaders class attribut.
            We need criterion(model(img_input), img_out_put). So the model(img_input) and
            the img_out_put need to be compatible in criterion i.e model(img_input).shape
            = img_out_put.shape ??(todo).

            Args:
            ----------
                opt_model : torch.optim.Optimizer
                    The optimization algorithm used for training.
                criterion : torch.nn.Module, optional
                    The loss function used for training (default torch.nn.CrossEntropyLoss()).
                num_epochs : int, optional
                    The number of epochs to train the model (default is 1).
        '''

        if num_epochs <= 0:
            raise ValueError('The num_epochs should be positive')
        
        size_data_set =len(self.data_loader.dataset)
        self.model.to(self.device)

        for epoch in range(num_epochs):
            loop = tqdm(enumerate(self.data_loader), total = len(self.data_loader))
            self.model.train() #* model in train mood
            train_MAE = 0
            for batch_idx, (img_input, img_out_put) in loop:
                img_input  =  img_input.to(self.device)
                img_out_put = img_out_put.to(self.device)
                opt_model.zero_grad()
                model_out_put = self.model(img_input)
                
                #* Get the batch loss and computing train MAE
                loss       = criterion(model_out_put, img_out_put)
                train_MAE += loss.item()*img_input.shape[0] #* img_input.shape[0] = self.batch_size, but the last batch could be diferente size

                #* Get gradients, and update the parameters of the model
                loss.backward()
                opt_model.step()

                #* Plot the loss and the progress bar
                loop.set_description(f"Epoch {epoch+1}/{num_epochs} process: {int((batch_idx / len(self.data_loader)) * 100)}")
                loop.set_postfix(modelLoss = loss.data.item())
            self.training_epochs += 1
            train_MAE = train_MAE / size_data_set
            print(f'Epoch completed, TRAIN MAE {train_MAE:.4f}')
            self.history["train_MAE"].append(train_MAE)

            if((self.data_set_val is not None) and get_val_MAE == True):
                #* We could try a diferent criterio for the val case in the same data_set.
                val_MAE = self.GetMAE(data_loader = self.data_loader_val, criterion = criterion)
                print(f'Epoch completed, VAL MAE: {(val_MAE):4f}')
                self.history["val_MAE"].append(val_MAE)

                #* Save the best model in val_MAE
                if(val_MAE >= min(self.history["val_MAE"])):
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict':opt_model.state_dict()
                    }, os.path.join(self.model_save_dir, f'checkpoint_epoch_{epoch + 1}_Val_MAE_{"{:.3f}".format(val_MAE)}.pt'))
            
            #* If we don have val_MAE, save in function of train_MAE
            else:
                if(train_MAE >= min(self.history["train_MAE"])):
                    torch.save({
                        'model_state_dict' : self.model.state_dict(),
                        'optimizer_state_dict' : opt_model.state_dict()
                    }, os.path.join(self.model_save_dir, f'checkpoint_epoch_{epoch + 1}_Train_MAE_{"{:.5f}".format(train_MAE)}.pt'))
    #TODO return the best model in MAE?

    def ToggleTrainingLayers(self, layers_list : list[str], enable : bool):
        '''
            This function will enable or disable the layers in the layers_list for the 
            training. Afther enable the layers we will print all the enable layers.

            Args:
            -----
                layers_list : list[str]
                    List with the layers name that we will enable for the training.
                enable : bool
                    True if the layer will be enable, false if the layer will be disable.
        '''

        for name, param in self.model.named_parameters():
            if any(layer_name in name for layer_name in layers_list):
                param.requires_grad = enable

        for name, param in self.model.named_parameters():
            if(param.requires_grad == True):
                print(f"{name} : {param.requires_grad}")

    def PrintHistorial(self, 
                        interval_train : list[int] = None, 
                        interval_validation : list[int] = None):
        '''
            Plot a img with the historial values that we have.

            Args:
                interval_train : list[int], optional
                    The interval of training epochs that we will plot.
                interval_validation : list[int], optional
                    The interval of training epochs that we will plot.
        '''

        if self.training_epochs == 0:
            print("self.training_epochs == 0 i.e model was not trained")
            return

        if interval_train is None:
            interval_train = [0, len(self.history['train_MAE'])]

        if interval_validation is None:
            interval_validation = [0, len(self.history['val_MAE'])]

        if interval_train[0] < 0 or interval_train[0] > self.training_epochs:
            raise ValueError('The interval_train[0] need to be in [0, training_epochs of the model)')
        if interval_train[1] > self.training_epochs:
            raise ValueError('The interval_train[1] need to be in [0, training_epochs of the model)')
        if interval_train[0] >= interval_train[1]:
            raise ValueError('We need interval_train[0] < interval_train[1]')
        if interval_validation[0] < 0 or interval_validation[0] > self.training_epochs:
            raise ValueError('The interval_validation[0] need to be in [0, training_epochs of the model)')
        if interval_validation[1] > self.training_epochs:
            raise ValueError('The interval_validation[1] need to be in [0, training_epochs of the model)')
        if interval_validation[0] >= interval_validation[1]:
            raise ValueError('We need interval_validation[0] < interval_validation[1]')

        epochs_values     = range(interval_train[0], interval_train[1])
        epochs_values_val = range(interval_validation[0], interval_validation[1])

        if(len(self.history['val_MAE']) != 0): #* Two img plots
            fig, (plt1) = plt.subplots(1, 1, figsize=(12, 6))
            plt1.plot(epochs_values,   self.history['train_MAE'][interval_train[0]: interval_train[1]], marker='o', color='blue', label='train MAE')
            plt1.set_xlabel('Epoch')
            plt1.set_title('Train MAE')     
            plt1.plot(epochs_values_val, self.history['val_MAE'][interval_validation[0]: interval_validation[1]], marker='o', color='red', label='validation MAE')
            plt1.set_xlabel('Epoch')
            plt1.set_title('Validation MAE Vs Train MAE')

            #* Add legend to each subplot
            plt1.legend()
            plt1.legend()
            #* Show the plots
            plt.show()

        elif(len(self.history['train_MAE']) != 0): #* One img plot
            fig, (plt1) = plt.subplots(1, 2, figsize=(12, 6))
            plt1.plot(epochs_values, self.history['train_MAE'][epochs_values[0] : epochs_values[-1]], marker='o', color='blue', label='MAE')
            plt1.set_xlabel('Epoch')
            plt1.set_ylabel('MAE')
            plt1.set_title('Train MAE')
            
            #* Add legend to each subplot
            plt.legend()
            #* Show the plots
            plt.show()
        
        else:
            print("len(self.history['val_MAE']) == 0, and \n len(self.history['train_MAE']) == 0")

    def Predict(self, index : int = 0, img_path : str = None):
        '''
            Use the model in a img, or in a attribute data_loader[index]

            Args:
            -----
                index : int
                    Index of the img in the data_loader
                img_path : str
                    Path of the image that we will use as model input.
        '''
        #TODO
        if(img_path is None):
            print("model(self.dataLoaders[index])")
        else:
            print("model_img_path")

    def GetDataBatch(self, index : int = 0):
        '''
            Get the a batch in data_loader_val for do testing.
            
            Args
            ----
                index : int = 0, optional
                    The index of the tensor batch in the data loader val.
            
            Returns
            -------
                Returns a data batch of the validation data set.
        '''

        for idx, (img_input, img_out_put) in enumerate(self.data_loader_val):
                if idx == index:
                    return img_input, img_out_put

        return None, None

class FiterUNet(fitertImgToImg):

    def __init__(self, 
                model: nn.Module, 
                data_set, 
                device: str = "cpu", 
                batch_size: int = 64, 
                data_set_val = None, 
                model_save_dir: str = None):
        super().__init__(model, data_set, device, batch_size, data_set_val, model_save_dir)

    def GetMAE(self,
            data_loader : DataLoader, 
            criterion : torch.nn.Module = torch.nn.CrossEntropyLoss(),
            ):
        '''
            Compute and return the MAE in 'data_loader' using 'criterion'.

            Args:
            -----
                data_loader : torch.utils.data.DataLoader  
                    index list of the batch tensors of the data_set
                criterion :  torch.nn.Module, optional
                    loss function of the model

            Returns
            -------
                Return a the value of MAE in 'data_loader' using 'criterion'.
        '''

        size_data_loader = len(data_loader.dataset)
        model_MAE = 0

        if size_data_loader == 0:
            raise ValueError('The data set should not be empty.')

        with torch.no_grad():
            for (img_input, img_out_put) in data_loader:
                img_input    =  img_input.to(self.device)
                img_out_put   = img_out_put.to(self.device, torch.long)
                model_out_put = self.model(img_input)

                model_out_put = model_out_put.view(model_out_put.shape[0], 2, -1)
                img_out_put = img_out_put.view(img_out_put.shape[0],  1, 68*68).squeeze(1)
                img_out_put = img_out_put.squeeze(1)
                loss = criterion(model_out_put, img_out_put)
                model_MAE += loss.item()*self.batch_size

        return model_MAE/size_data_loader


    def TrainModel(
            self,
            opt_model  : torch.optim, 
            criterion  : torch.nn.Module = torch.nn.CrossEntropyLoss(),
            num_epochs : int = 1,
            get_val_MAE  : bool = False,
        ):
        '''
            Funtion for train the model U-Net, we only change the line 
            loss = criterion(model_out_put, img_out_put) for the line
            loss = criterion(model_out_put[:,0,:,:],img_out_put[:,0,:,:])
        '''

        if num_epochs < 0:
            raise ValueError('num_epochs should be non-negative')
        
        size_data_set =len(self.data_loader.dataset)
        self.model.to(self.device)

        for epoch in range(num_epochs):
            loop = tqdm(enumerate(self.data_loader), total = len(self.data_loader))
            self.model.train() #* model in train mood
            train_MAE = 0
            for batch_idx, (img_input, img_out_put) in loop:
                img_input  =  img_input.to(self.device)
                img_out_put = img_out_put.to(self.device, torch.long)
                opt_model.zero_grad()
                model_out_put = self.model(img_input)

                #* Get the batch loss and computing train MAE
                model_out_put = model_out_put.view(model_out_put.shape[0], 2, -1)
                img_out_put = img_out_put.view(img_out_put.shape[0],  1, 68*68).squeeze(1)
                img_out_put = img_out_put.squeeze(1)
                loss       =  criterion(model_out_put, img_out_put)
                train_MAE += loss.item()*img_input.shape[0] #* img_input.shape[0] = self.batch_size, but the last batch could be diferente size

                #* Get gradients, and update the parameters of the model
                loss.backward()
                opt_model.step()

                #* Plot the loss and the progress bar
                loop.set_description(f"Epoch {epoch+1}/{num_epochs} process: {int((batch_idx / len(self.data_loader)) * 100)}")
                loop.set_postfix(modelLoss = loss.data.item())

            self.training_epochs += 1
            train_MAE = train_MAE / size_data_set
            print(f'Epoch completed, TRAIN MAE {train_MAE:.4f}')
            self.history["train_MAE"].append(train_MAE)

            if((self.data_set_val is not None) and get_val_MAE == True):
                val_MAE = self.GetMAE(data_loader = self.data_loader_val, criterion = criterion)
                #* We could try a diferent criterio for the val case in the same data_set.

                print(f'Epoch completed, VAL MAE: {(val_MAE):4f}')
                self.history["val_MAE"].append(val_MAE)

                #* Save the best model in val_MAE
                if(val_MAE >= min(self.history["val_MAE"])):
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict':opt_model.state_dict()
                    }, os.path.join(self.model_save_dir, f'checkpoint_epoch_{epoch + 1}_Val_MAE_{"{:.3f}".format(val_MAE)}.pt'))
            
            #* If we don have val_MAE, save in function of train_MAE
            else:
                if(train_MAE >= min(self.history["train_MAE"])):
                    torch.save({
                        'model_state_dict' : self.model.state_dict(),
                        'optimizer_state_dict' : opt_model.state_dict()
                    }, os.path.join(self.model_save_dir, f'checkpoint_epoch_{epoch + 1}_Train_MAE_{"{:.3f}".format(train_MAE)}.pt'))




