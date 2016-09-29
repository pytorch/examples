import torch.utils.data as data

from PIL import Image
import os
import os.path

import transforms


class ImageNetDataset(data.Dataset):
    def __init__(self, root, imgs, transform=None):
        self.root = root
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.imgs[index]

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs)


IMG_EXTENSIONS = [
    '.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG', '.ppm', '.PPM', '.bmp',
    '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    classes = os.listdir(dir)
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    for target in os.listdir(dir):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for filename in os.listdir(d):
            if is_image_file(filename):
                path = '{0}/{1}'.format(target, filename)
                item = (path, class_to_idx[target])
                images.append(item)

    return images


def make_datasets(root):
    traindir = os.path.join(root, 'train')
    valdir = os.path.join(root, 'val')

    classes, class_to_idx = find_classes(traindir)
    train_samples = make_dataset(traindir, class_to_idx)
    val_samples = make_dataset(valdir, class_to_idx)

    transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(),
    ])

    trainset = ImageNetDataset(traindir, train_samples, transform)
    valset = ImageNetDataset(valdir, val_samples, transform)

    return trainset, valset
