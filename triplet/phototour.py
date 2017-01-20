import os
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data


class PhotoTour(data.Dataset):
    #TODO: check this urls since I think that are not the same from the paper
    urls = [
        'http://phototour.cs.washington.edu/patches/trevi.zip',
        'http://phototour.cs.washington.edu/patches/notredame.zip',
        'http://phototour.cs.washington.edu/patches/halfdome.zip'
    ]
    image_ext = 'bmp'
    raw_folder = 'raw'
    processed_folder = 'processed'

    def __init__(self, root, name='notredame', transform=None, download=False):
        self.root = root
        self.transform = transform

        self.name = name
        self.fname = '{}.pt'.format(name)
        self.raw_dir = os.path.join(self.root, self.raw_folder, self.name)
        self.save_file = os.path.join(self.root, self.processed_folder,
                                      self.fname)

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.'
                               + ' You can use download=True to download it')

        print('Loading ...')

        self.data = torch.load(self.save_file)

        print('Done!')

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        if self.name == 'notredame':
            return 104192
        #TODO: check other sizes
        else:
            return 0

    def _check_exists(self):
        return os.path.exists(self.save_file)

    def download(self):
        if self._check_exists():
            print('Files already downloaded')
            return

        # TODO: implement download files routine

        # process and save as torch files
        print('Processing')

        data_set = (
            read_image_file(self.raw_dir, self.image_ext),
            # read_info_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        with open(self.save_file, 'wb') as f:
            torch.save(data_set, f)

        print('Done!')

def read_fnames(raw_dir, image_ext):
    """Return a list with the file names of the images containing the patches
    """
    files = []
    # find those files with the specified extension
    for file in os.listdir(raw_dir):
        if file.endswith(image_ext):
            files.append(os.path.join(raw_dir, file))
    return sorted(files)  # sort files in ascend order to keep relations

def read_image_file(raw_dir, image_ext):
    """Return a Tensor containing the patches
    """
    def PIL2array(img):
        return np.array(img.getdata(), np.uint8) \
            .reshape(img.size[1], img.size[0], 1)

    def read_fnames(raw_dir, image_ext):
        """Return a list with the file names of the images containing the patches
        """
        files = []
        # find those files with the specified extension
        for file in os.listdir(raw_dir):
            if file.endswith(image_ext):
                files.append(os.path.join(raw_dir, file))
        return sorted(files)  # sort files in ascend order to keep relations

    images = []
    list_files = read_fnames(raw_dir, image_ext)

    for file in list_files:
        assert os.path.isfile(file), 'Not a file: %s' % file
        # load the image containing the patches and convert to float point
        # and make sure that que only use one single channel
        img = PIL2array(Image.open(file))
        # split the image into patches and
        # add patches to buffer as individual elements
        patches_row = np.split(img, 16, axis=0)
        for row in patches_row:
            patches = np.split(row, 16, axis=1)
            for patch in patches:
                images.append(patch.reshape((64, 64)))
    return torch.ByteTensor(np.vstack(images)).view(-1, 64, 64)

if __name__ == '__main__':
    dataset = PhotoTour(root='/home/eriba/software/pytorch/examples-edgarriba/data',
                        name='notredame',
                        download=True)

    print dataset.data