import matplotlib.pyplot as plt
import numpy as np

def plot_img_tensor(tensor,
                    plot_channels : list[int] = [0,1,2],
                    title    : str = "my title",
                    localplt = None):
    '''
        Function for plot a img using a tensor. 
        The tensor shape need to be [Chanels, Height, Width], and
        plot_channels sub list of [0, Chanels]
        
        Args:
            tensor : torch.Tensor
                The img for plot in a tensor.
            plot_channels : list[int] 
                The channels for the plot
            localplt : plt.subplots, optional
                The enviroment to plot the tensor img
    '''

    if localplt is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    else:
        fig, ax = localplt

    if max(plot_channels) >= tensor.shape[0] or min(plot_channels) < 0:
        print("tensor.shape  = ", tensor.shape)
        print("plot_channels = ", plot_channels)
        raise ValueError('These channels are not in the tensor')

    image_array = tensor[plot_channels].detach().cpu().numpy()
    image_array = np.transpose(image_array, (1, 2, 0))

    ax.imshow(image_array)
    cbar = plt.colorbar(ax.imshow(image_array))
    cbar.set_label('Intensity')
    ax.axis('off')  # Turn off axis
    ax.set_title(title)

    if localplt is None:
        plt.show()