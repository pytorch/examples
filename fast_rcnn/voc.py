import torch
import torch.utils.data as data
from PIL import Image, ImageDraw
import os
import os.path
import sys
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


class TransformVOCDetectionAnnotation(object):
    def __init__(self, class_to_ind, keep_difficult=False):
        self.keep_difficult = keep_difficult
        self.class_to_ind = class_to_ind

    def __call__(self, target):
        #res = []
        #res = {}
        boxes = []
        gt_classes = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            #name = obj.find('name').text
            name = obj[0].text.lower().strip()
            #bb = obj.find('bndbox')
            bbox = obj[4]
            #bndbox = [bb.find('xmin').text, bb.find('ymin').text,
            #    bb.find('xmax').text, bb.find('ymax').text]
            # supposes the order is xmin, ymin, xmax, ymax
            # attention with indices
            bndbox = [int(bb.text)-1 for bb in bbox]

            #res += [bndbox + [name]]
            #res += [bndbox + [class_to_ind[name]]]
            boxes += [torch.LongTensor(bndbox)]
            gt_classes += [self.class_to_ind[name]]

        res = {
            'boxes': torch.cat([b.view(1,-1) for b in boxes], 0),
            'gt_classes':gt_classes
        }
        return res

class VOCSegmentation(data.Dataset):
    def __init__(self, root, image_set, transform=None, target_transform=None):
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform

        dataset_name = 'VOC2007'
        self._annopath = os.path.join(self.root, dataset_name, 'SegmentationClass', '%s.png')
        self._imgpath = os.path.join(self.root, dataset_name, 'JPEGImages', '%s.jpg')
        self._imgsetpath = os.path.join(self.root, dataset_name, 'ImageSets', 'Segmentation', '%s.txt')
 
        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip('\n') for x in self.ids]

    def __getitem__(self, index):
        img_id = self.ids[index]

        target = Image.open(self._annopath % img_id)#.convert('RGB')

        img = Image.open(self._imgpath % img_id).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.ids)


class VOCDetection(data.Dataset):
    def __init__(self, root, image_set, transform=None, target_transform=None):
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform

        dataset_name = 'VOC2007'
        self._annopath = os.path.join(self.root, dataset_name, 'Annotations', '%s.xml')
        self._imgpath = os.path.join(self.root, dataset_name, 'JPEGImages', '%s.jpg')
        self._imgsetpath = os.path.join(self.root, dataset_name, 'ImageSets', 'Main', '%s.txt')
 
        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip('\n') for x in self.ids]

    def __getitem__(self, index):
        img_id = self.ids[index]

        target = ET.parse(self._annopath % img_id).getroot()

        img = Image.open(self._imgpath % img_id).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.ids)

    def show(self, index):
        img, target = self.__getitem__(index)
        draw = ImageDraw.Draw(img)
        for obj in target:
            draw.rectangle(obj[0:4], outline=(255,0,0))
            draw.text(obj[0:2], obj[4], fill=(0,255,0))
        img.show()

if __name__ == '__main__':
    cls = ('__background__', # always index 0
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor')
    class_to_ind = dict(zip(cls, range(len(cls))))

    ds = VOCDetection('/home/francisco/work/datasets/VOCdevkit/', 'train',
            target_transform=TransformVOCDetectionAnnotation(class_to_ind, False))
    print(len(ds))
    img, target = ds[0]
    print(target)
    #ds.show(1)
    #dss = VOCSegmentation('/home/francisco/work/datasets/VOCdevkit/', 'train')
    #img, target = dss[0]

    #img.show()
    #print(target_transform(target))
