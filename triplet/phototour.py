import os
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data


class PhotoTour(data.Dataset):
    # TODO: check this urls since I think that are not the same from the paper
    urls = [
        'http://phototour.cs.washington.edu/patches/trevi.zip',
        'http://phototour.cs.washington.edu/patches/notredame.zip',
        'http://phototour.cs.washington.edu/patches/halfdome.zip'
    ]
    image_ext = 'bmp'
    info_file = 'info.txt'
    matches_files = 'm50_100000_100000.txt'

    def __init__(self, root, name='notredame', transform=None, download=False, size=64):
        self.root = root
        self.size = size
        self.transform = transform

        self.name = name
        self.data_dir = os.path.join(self.root, name)
        self.data_file = os.path.join(self.root, '{}.pt'.format(name))

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.'
                               + ' You can use download=True to download it')

        # load the serialized data
        self.data, self.labels, self.matches = torch.load(self.data_file)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        if self.name == 'notredame':
            return 104192
        # TODO: check other sizes
        else:
            return 0

    def _check_exists(self):
        return os.path.exists(self.data_file)

    def download(self):
        print('Loading PhotoTour dataset: {}'.format(self.name))

        if self._check_exists():
            print('Files already downloaded!')
            return

        # TODO: implement download files routine

        # process and save as torch files
        print('Processing ...')

        data_set = (
            read_image_file(self.data_dir, self.image_ext, self.size),
            read_info_file(self.data_dir, self.info_file),
            read_matches_files(self.data_dir, self.matches_files)
        )
        with open(self.data_file, 'wb') as f:
            torch.save(data_set, f)

        print('Done!')


def read_image_file(data_dir, image_ext, img_sz):
    """Return a Tensor containing the patches
    """
    def PIL2array(_img, img_size):
        """Convert PIL image type to numpy 2D array
        """
        return np.array(_img.getdata(), dtype=np.uint8) \
            .reshape(img_size, img_size)

    def read_filenames(_data_dir, _image_ext):
        """Return a list with the file names of the images containing the patches
        """
        files = []
        # find those files with the specified extension
        for file_dir in os.listdir(_data_dir):
            if file_dir.endswith(_image_ext):
                files.append(os.path.join(_data_dir, file_dir))
        return sorted(files)  # sort files in ascend order to keep relations

    images = []
    list_files = read_filenames(data_dir, image_ext)

    for file_path in list_files:
        assert os.path.isfile(file_path), 'Not a file: %s' % file_path
        # load the image containing the patches, crop in 64x64 patches and
        # reshape to the desired size (default: 64)
        img = Image.open(file_path)
        for y in range(0, 1024 - 64, 64):
            for x in range(0, 1024 - 64, 64):
                patch = img.crop((x, y, x + 64, y + 64))
                if img_sz != 64:
                    patch = patch.resize((img_sz, img_sz), Image.BICUBIC)
                images.append(PIL2array(patch, img_sz))
    return torch.ByteTensor(np.vstack(images)).view(-1, img_sz, img_sz)


def read_info_file(data_dir, info_file):
    return 0


def read_matches_files(data_dir, matches_file):
    return 0


if __name__ == '__main__':
    dataset = PhotoTour(root='/home/eriba/datasets/patches_dataset',
                        name='notredame',
                        download=True,
                        size=32)

    print('Loaded PhotoTour: {} with {} images.'
          .format(dataset.name, len(dataset.data)))
