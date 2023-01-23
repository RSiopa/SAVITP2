#!/usr/bin/env python3

import argparse
import random
from statistics import mean
import cv2
from colorama import Fore, Style
from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import os
from classification_visualizer import ClassificationVisualizer
from data_visualizer import DataVisualizer
from model import Model
from dataset import Dataset
import glob
import numpy as np


def main():

    # -----------------------------------------------------
    # Initialization
    # -----------------------------------------------------

    # Setting up the arguments
    parser = argparse.ArgumentParser(description='Training of the Deep Learning Model')  # arguments
    parser.add_argument('-rt', '--resume_training', action='store_true', default=True, 
                        help='Continue previous training.\n ')
    parser.add_argument('-lr', '--learning_rate', default=0.001, 
                        help='Define the learning rate.\n ')
    parser.add_argument('-mne', '--maximum_num_epochs', type=int, default=1000, 
                        help='Define the maximum number of epochs.\n ')
    parser.add_argument('-tlt', '--termination_loss_threshold', default=0.001, 
                        help='Define the termination loss threshold.\n ')
    parser.add_argument('-mp', '--model_path', default='model.pkl', 
                        help='Define the path for the model file.\n ')
    parser.add_argument('-dp', '--dataset_path', default='Datasets/rgbd-dataset', 
                        help='Define the path for the dataset.\n ')
    parser.add_argument('-in', '--image_number', type=int, choices=range(1, 200000), default=1000, 
                        help='Define the number of images to use for sampling.\n ')
    parser.add_argument('-ts', '--test_size', default=0.2, 
                        help='Define the percentage of images used for testing.\n ')
    parser.add_argument('-tbs', '--train_batch_size', type=int, choices=range(25, 10000), default=500, 
                        help='Define batch size for training.\n ')
    parser.add_argument('-tesbs', '--test_batch_size', type=int, choices=range(25, 10000), default=500, 
                        help='Define batch size for testing.\n ')
    args = vars(parser.parse_args())

    resume_training = args['resume_training']
    model_path = args['model_path']
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = Model()
    model.to(device)

    learning_rate = args['learning_rate']
    maximum_num_epochs = args['maximum_num_epochs']
    termination_loss_threshold = args['termination_loss_threshold']
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    # Create the dataset
    dataset_path = args['dataset_path']

    # Get all the images from the dataset
    image_filenames = []
    mask_filenames = []
    dir_list = os.listdir(dataset_path)
    for dirs in dir_list:
        subdir_list = os.listdir(dataset_path + '/' + dirs)
        # print(subdir_list)
        for subdirs in subdir_list:
            file_list = os.listdir(dataset_path + '/' + dirs + '/' + subdirs)
            # print(file_list)
            for file in file_list:
                # print(file)
                if file.endswith('_crop.png') and os.path.isfile((dataset_path + '/' + dirs + '/' + subdirs + '/' + file).replace("_crop.png", "_maskcrop.png")):
                    image_filenames.append(dataset_path + '/' + dirs + '/' + subdirs + '/' + file)

    # Chose random sample out of aproximately 200000 images
    image_filenames = random.sample(image_filenames, k=args['image_number'])

    # Split images to train from images to test
    train_image_filenames, test_image_filenames = train_test_split(image_filenames, test_size=args['test_size'])

    # Train dataset
    dataset_train = Dataset(train_image_filenames)
    loader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=args['train_batch_size'], shuffle=True)

    # Test dataset
    dataset_test = Dataset(test_image_filenames)
    loader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=args['test_batch_size'], shuffle=True)

    tensor_to_pil_image = transforms.ToPILImage()

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    loss_visualizer = DataVisualizer('Loss')
    loss_visualizer.draw([0, maximum_num_epochs], [termination_loss_threshold, termination_loss_threshold], layer='threshold', marker='--', markersize=1, color=[0.5, 0.5, 0.5], alpha=1, label='threshold', x_label='Epochs', y_label='Loss')

    test_visualizer = ClassificationVisualizer('Test Images')

    # Resume training
    if resume_training:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        idx_epoch = checkpoint['epoch']
        epoch_train_losses = checkpoint['train_losses']
        epoch_test_losses = checkpoint['test_losses']     
    else:
        idx_epoch = 0
        epoch_train_losses = []
        epoch_test_losses = []

    # Move the model variable to the gpu if one exists
    model.to(device)  

    while True:
        

        # Train

        train_losses = []
        for batch_idx, (image_t, label_t) in tqdm(enumerate(loader_train), total=len(loader_train), desc=Fore.GREEN + 'Training batches for Epoch ' + str(idx_epoch) + Style.RESET_ALL):

            image_t = image_t.to(device)
            label_t = label_t.to(device)

            # Get output from the model, given the inputs
            label_t_predicted = model.forward(image_t)

            # Get loss for the predicted output
            loss = loss_function(label_t_predicted, label_t)

            # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
            optimizer.zero_grad()

            # Get gradients w.r.t to parameters
            loss.backward()

            # Update parameters
            optimizer.step()

            train_losses.append(loss.data.item())

        # Compute the loss for the epoch
        epoch_train_loss = mean(train_losses)
        epoch_train_losses.append(epoch_train_loss)


        # Test

        test_losses = []
        for batch_idx, (image_t, label_t) in tqdm(enumerate(loader_test), total=len(loader_test),
                                                  desc=Fore.GREEN + 'Testing batches for Epoch ' + str(
                                                          idx_epoch) + Style.RESET_ALL):
            image_t = image_t.to(device)
            label_t = label_t.to(device)

            # Apply the network to get the predicted label
            label_t_predicted = model.forward(image_t)

            # Compute the error based on the predictions
            loss = loss_function(label_t_predicted, label_t)

            test_losses.append(loss.data.item())

            test_visualizer.draw(image_t, label_t, label_t_predicted)

        # Compute the loss for the epoch
        epoch_test_loss = mean(test_losses)
        epoch_test_losses.append(epoch_test_loss)

        # -----------------------------------------------------
        # Visualization
        # -----------------------------------------------------

        loss_visualizer.draw(list(range(0, len(epoch_train_losses))), epoch_train_losses, layer='train loss', marker='-', markersize=1, color=[0,0,0.7], alpha=1, label='Train Loss', x_label='Epochs', y_label='Loss')

        loss_visualizer.draw(list(range(0, len(epoch_test_losses))), epoch_test_losses, layer='test loss', marker='-',
                             markersize=1, color=[1, 0, 0.7], alpha=1, label='Test Loss', x_label='Epochs',
                             y_label='Loss')

        loss_visualizer.recomputeAxesRanges()

        print(Fore.BLUE + 'Epoch ' + str(idx_epoch) + ' Loss ' + str(epoch_train_loss) + Style.RESET_ALL)

        # Save checkpoint
        model.to('cpu')
        torch.save({
            'epoch': idx_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': epoch_train_losses,
            'test_losses': epoch_test_losses,
        }, model_path)
        model.to(device)

        # Go to next epoch
        idx_epoch += 1  

        # Termination criteria
        if idx_epoch > maximum_num_epochs:
            print('Finished training. Reached maximum number of epochs.')
            break
        elif epoch_train_loss < termination_loss_threshold:
            print('Finished training. Reached target loss.')
            break
    
    # End result until key pressed
    key = plt.waitforbuttonpress(0)
    exit(0)


if __name__ == "__main__":
    main()
