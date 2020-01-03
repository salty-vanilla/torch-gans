import functools
from torchvision import datasets, transforms
import torch
from PIL import Image
import os
import numpy as np


class _BaseDataset():
    def __init__(self, target_size=None,
                 is_flip=False,
                 *args, **kwargs):
        self.transform = []
        if target_size is not None:
            if not isinstance(target_size, int):
                target_size = (target_size[1], target_size[0])
            self.transform.append(transforms.Resize(target_size))
        if is_flip:
            self.transform.append(transforms.RandomHorizontalFlip())
        self.transform.append(transforms.ToTensor())
        self.transform.append(transforms.Lambda(lambda x: (x - 0.5)*2))
        self.transform = transforms.Compose(self.transform)


class ArrayDataset(_BaseDataset):
    def __init__(self, x,
                 y=None,
                 target_size=None, 
                 is_flip=False,
                 *args, **kwargs):
        super().__init__(target_size, is_flip)
        self.x = x
        self.y = y

    def __getitem__(index):
        image = Image.fromarray(self.x[index])
        image = self.transform(image)

        if self.y:
            return image, self.y[index]
        else:
            return image

    def __len__(self):
        return len(self.x)


class PathDataset(_BaseDataset):
    def __init__(self, paths,
                 labels=None,
                 target_size=None, 
                 is_flip=False,
                 *args, **kwargs):
        super().__init__(target_size, is_flip)
        self.paths = path
        self.labels = labels

    def __getitem__(self, index):
        path = self.paths[index]
        image = Image.open(path)
        image = self.transform(image)

        if self.labels:
            return image, self.labels[index]
        
        else:
            return image

    def __len__(self):
        return len(self.paths)


class DirectoryDataset(PathDataset):
    def __init__(self, image_dir,
                 with_labels=False,
                 target_size=None, 
                 is_flip=False,
                 *args, **kwargs):
        if with_labels:
            dirs = [os.path.join(image_dir, f)
                    for f in os.listdir(image_dir)
                    if os.path.isdir(os.path.join(image_dir, f))]
            dirs = sorted(dirs)
            image_paths = [get_image_paths(d) for d in dirs]
            labels = []
            for i, ip in enumerate(image_paths):
                labels += [i] * len(ip)
            labels = np.array(labels)
            image_paths = np.array(functools.reduce(lambda x, y: x+y, image_paths))
        else:
            image_paths = np.array([path for path in get_image_paths(image_dir)])
            labels = None

        super().__init__(image_paths, 
                         labels,
                         target_size,
                         is_flip)


def get_image_paths(src_dir):
    def get_all_paths():
        for root, dirs, files in os.walk(src_dir):
            yield root
            for file in files:
                yield os.path.join(root, file)

    def is_image(path):
        _, ext = os.path.splitext(path)
        ext = ext[1:]
        if ext in ['png', 'jpg', 'bmp', 'PNG', 'JPG', 'BMP']:
            return True
        else:
            return False

    return [path for path in get_all_paths() if is_image(path)]
