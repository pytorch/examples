import torch
import math
import random
from PIL import Image


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class ToTensor(object):
    def __call__(self, pic):
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        img = img.view(pic.size[0], pic.size[1], 3)
        # put it in CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 2).transpose(1, 2).contiguous()
        return img.float()

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor


class Scale(object):
    "Scales the smaller edge to size"
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return img
        if w < h:
            return img.resize((w, int(round(h / w * self.size))), self.interpolation)
        else:
            return img.resize((int(round(w / h * self.size)), h), self.interpolation)


class CenterCrop(object):
    "Crop to centered rectangle"
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        w, h = img.size
        x1 = int(round((w - self.size) / 2))
        y1 = int(round((h - self.size) / 2))
        return img.crop((x1, y1, x1 + self.size, y1 + self.size))


class RandomCrop(object):
    "Random crop form larger image with optional zero padding"
    def __init__(self, size, padding=0):
        self.size = size
        self.padding = padding

    def __call__(self, img):
        if self.padding > 0:
            raise NotImplementedError()

        w, h = img.size
        if w == self.size and h == self.size:
            return img

        x1 = random.randint(0, w - self.size)
        y1 = random.randint(0, h - self.size)
        return img.crop((x1, y1, x1 + self.size, y1 + self.size))


class RandomHorizontalFlip(object):
    "Horizontal flip with 0.5 probability"
    def __call__(self, img):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class RandomSizedCrop(object):
    "Random crop with size 0.08-1 and aspect ratio 3/4 - 4/3 (Inception-style)"
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3 / 4, 4 / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))

                return img.resize((self.size, self.size), self.interpolation)

        # Fallback
        scale = Scale(self.size, interpolation=self.interpolation)
        crop = CenterCrop(self.size)
        return crop(scale(img))
