import torch
from PIL import Image
from torchvision.transforms import transforms
import numpy as np


class Dataset(torch.utils.data.Dataset):

    def __init__(self, image_filenames):

        super().__init__()

        self.image_filenames = image_filenames
        self.num_images = len(self.image_filenames)

        self.labels = []
        for image_filename in self.image_filenames:
            self.labels.append(self.getClassFromFilename(image_filename))

        # Create a set of transformations
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()])

    def __getitem__(self, index):   # Return specific element x, y given the index, of the dataset

        # Get the image
        image_pil = Image.open(self.image_filenames[index])

        image_t = self.transforms(image_pil)

        return image_t, self.labels[index]

    def __len__(self):  # Return the length of the dataset
        return self.num_images

    def getClassFromFilename(self, filename):

        parts = filename.split('/')
        part = parts[-1]

        parts = part.split('.')
        parts = part.split('_')

        class_name = parts[0]
        # print(type(parts[1]))
        if not parts[1].isnumeric():
            class_name = class_name + ' ' + parts[1]

        # print('filename ' + filename + ' is a ' + class_name)

        if class_name == 'apple':
            label = 0
        elif class_name == 'ball':
            label = 1
        elif class_name == 'banana':
            label = 2
        elif class_name == 'bell pepper':
            label = 3
        elif class_name == 'binder':
            label = 4
        elif class_name == 'bowl':
            label = 5
        elif class_name == 'calculator':
            label = 6
        elif class_name == 'camera':
            label = 7
        elif class_name == 'cap':
            label = 8
        elif class_name == 'cell phone':
            label = 9
        elif class_name == 'cereal box':
            label = 10
        elif class_name == 'coffee mug':
            label = 11
        elif class_name == 'comb':
            label = 12
        elif class_name == 'dry battery':
            label = 13
        elif class_name == 'flashlight':
            label = 14
        elif class_name == 'food bag':
            label = 15
        elif class_name == 'food box':
            label = 16
        elif class_name == 'food can':
            label = 17
        elif class_name == 'food cup':
            label = 18
        elif class_name == 'food jar':
            label = 19
        elif class_name == 'garlic':
            label = 20
        elif class_name == 'glue stick':
            label = 21
        elif class_name == 'greens':
            label = 22
        elif class_name == 'hand towel':
            label = 23
        elif class_name == 'instant noodles':
            label = 24
        elif class_name == 'keyboard':
            label = 25
        elif class_name == 'kleenex':
            label = 26
        elif class_name == 'lemon':
            label = 27
        elif class_name == 'lightbulb':
            label = 28
        elif class_name == 'lime':
            label = 29
        elif class_name == 'marker':
            label = 30
        elif class_name == 'mushroom':
            label = 31
        elif class_name == 'notebook':
            label = 32
        elif class_name == 'onion':
            label = 33
        elif class_name == 'orange':
            label = 34
        elif class_name == 'peach':
            label = 35
        elif class_name == 'pear':
            label = 36
        elif class_name == 'pitcher':
            label = 37
        elif class_name == 'plate':
            label = 38
        elif class_name == 'pliers':
            label = 39
        elif class_name == 'potato':
            label = 40
        elif class_name == 'rubber eraser':
            label = 41
        elif class_name == 'scissors':
            label = 42
        elif class_name == 'shampoo':
            label = 43
        elif class_name == 'soda can':
            label = 44
        elif class_name == 'sponge':
            label = 45
        elif class_name == 'stapler':
            label = 46
        elif class_name == 'tomato':
            label = 47
        elif class_name == 'toothbrush':
            label = 48
        elif class_name == 'toothpaste':
            label = 49
        elif class_name == 'water bottle':
            label = 50
        else:
            raise ValueError('Unknown class')

        return label
