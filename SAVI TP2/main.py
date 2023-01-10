#!/usr/bin/env python3

#test
import random
from statistics import mean

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


def main():

    # -----------------------------------------------------
    # Initialization
    # -----------------------------------------------------

    resume_training = False
    model_path = 'model.pkl'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = Model()
    model.to(device)

    learning_rate = 0.001
    maximum_num_epochs = 1000
    termination_loss_threshold = 0.001
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    # Create the dataset
    dataset_path = 'Datasets/rgbd-dataset'

    # Get all the images from the dataset
    image_filenames = []
    dir_list = os.listdir(dataset_path)
    for dirs in dir_list:
        subdir_list = os.listdir(dataset_path + '/' + dirs)
        # print(subdir_list)
        for subdirs in subdir_list:
            file_list = os.listdir(dataset_path + '/' + dirs + '/' + subdirs)
            # print(file_list)
            for file in file_list:
                # print(file)
                if file.endswith('_crop.png'):
                    image_filenames.append(dataset_path + '/' + dirs + '/' + subdirs + '/' + file)
                    # print(image_filenames)
                    # print(dataset_path + '/' + dirs + '/' + subdirs + '/' + file)
            # print(len(image_filenames))
            # print(image_filenames)

    # print(len(image_filenames))  # 207920 images
    image_filenames = random.sample(image_filenames, k=199)
    # print(image_filenames)

    train_image_filenames, test_image_filenames = train_test_split(image_filenames, test_size=0.2)

    dataset_train = Dataset(train_image_filenames)
    loader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=10, shuffle=True)

    dataset_test = Dataset(test_image_filenames)
    loader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=10, shuffle=True)

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

        # model.train()
    else:
        idx_epoch = 0
        epoch_train_losses = []
        epoch_test_losses = []
    # -----------

    model.to(device)  # move the model variable to the gpu if one exists

    while True:

        train_losses = []
        for batch_idx, (image_t, label_t) in tqdm(enumerate(loader_train), total=len(loader_train), desc=Fore.GREEN + 'Training batches for Epoch ' + str(idx_epoch) + Style.RESET_ALL):

            image_t = image_t.to(device)
            label_t = label_t.to(device)

            # get output from the model, given the inputs
            label_t_predicted = model.forward(image_t)
            print(label_t)
            print(label_t_predicted)

            # get loss for the predicted output
            loss = loss_function(label_t_predicted, label_t)

            # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
            optimizer.zero_grad()

            # get gradients w.r.t to parameters
            loss.backward()

            # update parameters
            optimizer.step()

            train_losses.append(loss.data.item())

        epoch_train_loss = mean(train_losses)
        epoch_train_losses.append(epoch_train_loss)

        test_losses = []
        for batch_idx, (image_t, label_t) in tqdm(enumerate(loader_test), total=len(loader_test),
                                                  desc=Fore.GREEN + 'Testing batches for Epoch ' + str(
                                                          idx_epoch) + Style.RESET_ALL):
            image_t = image_t.to(device)
            label_t = label_t.to(device)

            # Apply the network to get the predicted ys
            label_t_predicted = model.forward(image_t)

            # Compute the error based on the predictions
            loss = loss_function(label_t_predicted, label_t)

            test_losses.append(loss.data.item())

            test_visualizer.draw(image_t, label_t, label_t_predicted)

        # Compute the loss for the epoch
        epoch_test_loss = mean(test_losses)
        epoch_test_losses.append(epoch_test_loss)

        # Visualization
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

        idx_epoch += 1  # go to next epoch
        # Termination criteria
        if idx_epoch > maximum_num_epochs:
            print('Finished training. Reached maximum number of epochs.')
            break
        elif epoch_train_loss < termination_loss_threshold:
            print('Finished training. Reached target loss.')
            break

    key = plt.waitforbuttonpress(0)
    exit(0)


if __name__ == "__main__":
    main()
