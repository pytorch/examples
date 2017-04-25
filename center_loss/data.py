from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
import random

class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)

def get_dataset(paths):
    dataset = []
    for path in paths.split(':'):
        path_exp = os.path.expanduser(path)
        classes = os.listdir(path_exp)
        classes.sort()
        nrof_classes = len(classes)
        for i in range(1000):
            class_name = classes[i]
            facedir = os.path.join(path_exp, class_name)
            if os.path.isdir(facedir):
                images = os.listdir(facedir)
                image_paths = [os.path.join(facedir,img) for img in images]
                dataset.append(ImageClass(class_name, image_paths))

    return dataset, classes

def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths)

    return image_paths_flat, labels_flat

def split_train_test(image_paths, labels, split=0.1):
    train_indices = []
    val_indices = []
    for label in range(len(set(labels))):
        indexes = [i for i,x in enumerate(labels) if x == label]
        val_indices.extend(indexes[:int(np.floor(len(indexes)*split))])
        train_indices.extend(indexes[int(np.floor(len(indexes)*split)):])

    selected_val = [(image_paths[i], labels[i]) for i in sorted(val_indices)]
    selected_train = [(image_paths[i], labels[i]) for i in sorted(train_indices)]
    val_image_paths, val_labels = tuple([list(tup) for tup in zip(*selected_val)])
    train_image_paths, train_labels = tuple([list(tup) for tup in zip(*selected_train)])

    assert len(set(val_labels) - set(train_labels)) == 0, "validation labels should be a subset of the train labels"

    return train_image_paths, train_labels, val_image_paths, val_labels

def default_loader(path):
    return Image.open(path).convert('RGB')

class ImageDataset(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None,
                 train=True, split=0.1, loader=default_loader):
        self.root = root
        self.train = train
        self.split = split
        self.dataset, self.classes = get_dataset(self.root)
        image_paths, labels = get_image_paths_and_labels(self.dataset)
        self.train_image_paths, self.train_labels, self.val_image_paths, self.val_labels = split_train_test(image_paths, labels, self.split)
        if len(self.dataset) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + self.root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader


    def __getitem__(self, index):
        if self.train:
            path, target = self.train_image_paths[index], self.train_labels[index]
        else:
            path, target = self.val_image_paths[index], self.val_labels[index]

        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_image_paths)
        else:
            return len(self.val_image_paths)
