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
    mean = {'notredame': 0.4854, 'yosemite': 0.4844, 'liberty': 0.4437}
    std = {'notredame': 0.1864, 'yosemite': 0.1818, 'liberty': 0.2019}
    lens = {'notredame': 468159, 'yosemite': 633587, 'liberty': 450092}

    image_ext = 'bmp'
    info_file = 'info.txt'
    matches_files = 'm50_100000_100000_0.txt'

    def __init__(self, root, name='notredame', transform=None,
                 download=False, size=64):
        self.root = root
        self.size = size
        self.transform = transform

        self.name = name
        self.data_dir = os.path.join(self.root, name)
        self.data_file = os.path.join(self.root, '{}_{}.pt'.format(name, size))

        self.mean = self.mean[name]
        self.std = self.std[name]

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
        return self.lens[self.name]

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
            read_image_file(self.data_dir, self.image_ext, self.size, self.__len__()),
            read_info_file(self.data_dir, self.info_file),
            read_matches_files(self.data_dir, self.matches_files)
        )

        with open(self.data_file, 'wb') as f:
            torch.save(data_set, f)

        print('Done!')


def read_image_file(data_dir, image_ext, img_sz, n):
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
        # load the image containing the patches, crop in 64x64 patches and
        # reshape to the desired size (default: 64)
        img = Image.open(file_path)
        for y in range(0, 1024, 64):
            for x in range(0, 1024, 64):
                patch = img.crop((x, y, x + 64, y + 64))
                if img_sz != 64:
                    patch = patch.resize((img_sz, img_sz), Image.BICUBIC)
                images.append(PIL2array(patch, img_sz))
    return torch.ByteTensor(np.vstack(images)).view(-1, img_sz, img_sz)[:n]


def read_info_file(data_dir, info_file):
    """Return a Tensor containing the list of labels
       Read the file and keep only the ID of the 3D point.
    """
    labels = []
    with open(os.path.join(data_dir, info_file), 'r') as f:
        for line in f:
            labels.append(int(line.split()[0]))
    return np.array(labels)


def read_matches_files(data_dir, matches_file):
    """Return a Tensor containing the ground truth matches
       Read the file and keep only 3D point ID.
       Matches are represented with a 1, non matches with a 0.
    """
    matches = []
    with open(os.path.join(data_dir, matches_file), 'r') as f:
        for line in f:
            l = line.split()
            matches.append([int(l[0]), int(l[3]), int(l[1] == l[4])])
    return np.array(matches)


if __name__ == '__main__':
    dataset = PhotoTour(root='/home/eriba/datasets/patches_dataset',
                        name='yosemite',
                        download=True,
                        size=32)

    print('Loaded PhotoTour: {} with {} images.'
          .format(dataset.name, len(dataset.data)))

    assert len(dataset.data) == len(dataset.labels)
