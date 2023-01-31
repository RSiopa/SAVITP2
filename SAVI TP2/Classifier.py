#!/usr/bin/python3

import argparse
import os
import glob
import random
from statistics import mean
import sys
from sklearn.model_selection import train_test_split
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from colorama import Fore, Style
import torch
from PIL import Image
from model import Model


# Image Classifier
def Classifier(image):

    # Use best model
    model_path = 'Best_Model/model.pkl'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # cuda: 0 index of gpu
    model = Model()

    load_model = torch.load(model_path)
    model.load_state_dict(load_model['model_state_dict'])
    model.to(device)
    model.eval()

    PIL_to_Tensor = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    
    # PIL to Tensor
    image_t= PIL_to_Tensor(image)
    image_t = image_t.unsqueeze(0)

    image_t = image_t.to(device)
    label_t_predicted = model.forward(image_t)
    
    prediction = torch.argmax(label_t_predicted)
    label = prediction.data.item()

    # Every possible label
    if label == 0:
        class_name = 'apple'
    elif label == 1:
        class_name = 'ball'
    elif label == 2:
        class_name = 'banana'
    elif label == 3:
        class_name = 'bell_pepper'
    elif label == 4:
        class_name = 'binder'
    elif label == 5:
        class_name = 'bowl'
    elif label == 6:
        class_name = 'calculator'
    elif label == 7:
        class_name = 'camera'
    elif label == 8:
        class_name = 'cap'
    elif label == 9:
        class_name = 'cell_phone'
    elif label == 10:
        class_name = 'cereal_box'
    elif label == 11:
        class_name = 'coffee_mug'
    elif label == 12:
        class_name = 'comb'
    elif label == 13:
        class_name = 'dry_battery'
    elif label == 14:
        class_name = 'flashlight'
    elif label == 15:
        class_name = 'food_bag'
    elif label == 16:
        class_name = 'food_box'
    elif label == 17:
        class_name = 'food_can'
    elif label == 18:
        class_name = 'food_cup'
    elif label == 19:
        class_name = 'food_jaar'
    elif label == 20:
        class_name = 'garlic'
    elif label == 21:
        class_name = 'glue_stick'
    elif label == 22:
        class_name = 'greens'
    elif label == 23:
        class_name = 'hand_towel'
    elif label == 24:
        class_name = 'instant_noodles'
    elif label == 25:
        class_name = 'keyboard'
    elif label == 26:
        class_name = 'kleenex'
    elif label == 27:
        class_name = 'lemon'
    elif label == 28:
        class_name = 'lightbulb'
    elif label == 29:
        class_name = 'lime'
    elif label == 30:
        class_name = 'marker'
    elif label == 31:
        class_name = 'mushroom'
    elif label == 32:
        class_name = 'notebook'
    elif label == 33:
        class_name = 'onion'
    elif label == 34:
        class_name = 'orange'
    elif label == 35:
        class_name = 'peach'
    elif label == 36:
        class_name = 'pear'
    elif label == 37:
        class_name = 'pitcher'
    elif label == 38:
        class_name = 'plate'
    elif label == 39:
        class_name = 'pliers'
    elif label == 40:
        class_name = 'potato'
    elif label == 41:
        class_name = 'rubber_eraser'
    elif label == 42:
        class_name = 'scissors'
    elif label == 43:
        class_name = 'shampoo'
    elif label == 44:
        class_name = 'soda_can'
    elif label == 45:
        class_name = 'sponge'
    elif label == 46:
        class_name = 'stapler'
    elif label == 47:
        class_name = 'tomato'
    elif label == 48:
        class_name = 'toothbrush'
    elif label == 49:
        class_name = 'toothpaste'
    elif label == 50:
        class_name = 'water_bottle'
    else:
        raise ValueError('Unknown class')

    return class_name