import pickle
import os
import pandas as pd
from tqdm import tqdm
#from src.plotting import *
#from src.utils import *
#from src.diagnosis import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from time import time
from pprint import pprint
import os
import torch
import numpy as np
# from beepy import beep

def convert_to_windows(data, model,config):
	windows = []; w_size = model.n_window
	for i, g in enumerate(data):
		if i >= w_size: w = data[i-w_size:i]
		else: w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
		windows.append(w if 'TransNAS_TSAD' in config.model else w.view(-1))
	return torch.stack(windows)

def save_model(model, optimizer, scheduler, epoch, accuracy_list,config):
	folder = f'TransNAS_checkpoints/{config.model}_{config.dataset}/'
	os.makedirs(folder, exist_ok=True)
	file_path = f'{folder}/model.ckpt'
	torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)
 


def load_model(modelname, dims,config, **kwargs):
    """
    Loads or initializes a model along with its optimizer and scheduler based on provided hyperparameters.

    Args:
        modelname (str): The name of the model class to be loaded from the `src.models` module.
        dims (int): The dimensionality of the input data, typically the number of features.
        **kwargs: Arbitrary keyword arguments containing hyperparameters for model initialization.

    Returns:
        tuple: A tuple containing the model, optimizer, scheduler, the last epoch trained, and a list of accuracy metrics.
    """
    import src.models
    # Dynamically import the model class based on the given model name
    model_class = getattr(src.models, modelname)
    print("Hyperparameters:", kwargs)

    # Initialize the model with provided dimensions and hyperparameters
    model = model_class(dims, **kwargs)

    # Initialize the optimizer with the learning rate and weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=kwargs.get('lr', model.lr), weight_decay=1e-5)
    # Initialize the scheduler with step size and gamma for learning rate decay
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 15, 0.9)

    # Define the filename for saving/loading the model checkpoint
    fname = f'TransNAS_checkpoints/{config.model}_{config.dataset}/model.ckpt'

    # Load the model and optimizer states if a checkpoint exists and retraining/testing is not explicitly requested
    if os.path.exists(fname) and (not config.retrain or config.test):
        print(f"Loading pre-trained model: {model.name}")
        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        accuracy_list = checkpoint['accuracy_list']
    else:
        # Otherwise, indicate that a new model is being created
        print(f"Creating new model: {model.name}")
        epoch = -1  # Start from epoch -1 to indicate no prior training
        accuracy_list = []  # Initialize an empty list for tracking accuracy metrics

    return model, optimizer, scheduler, epoch, accuracy_list


def load_dataset(path,dataset,config):
	folder = os.path.join(path, dataset)
	if not os.path.exists(folder):
		raise Exception('Processed Data not found.')
	loader = []
	for file in ['train', 'test', 'labels']:
		if dataset == 'SMD': file = 'machine-1-1_' + file
		if dataset == 'SMAP': file = 'P-1_' + file
		if dataset == 'MSL': file = 'C-1_' + file
		if dataset == 'UCR': file = '136_' + file
		if dataset == 'NAB': file = 'ec2_request_latency_system_failure_' + file
		loader.append(np.load(os.path.join(folder, f'{file}.npy')))
	# loader = [i[:, debug:debug+1] for i in loader]
	#if config.less: loader[0] = cut_array(0.2, loader[0])
	train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
	test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
	labels = loader[2]
	return train_loader, test_loader, labels

def optimize_model(epoch, model, data, dataO, optimizer, scheduler,config, training=True):
    """
    This function optimizes a given model for a single epoch, either in training or evaluation mode.

    Parameters:
    - epoch (int): Current epoch number.
    - model (torch.nn.Module): The model to be optimized.
    - data (array-like): Input data for the model.
    - dataO (array-like): The actual outputs/targets for the input data.
    - optimizer (torch.optim.Optimizer): Optimizer used for model parameter updates.
    - scheduler (torch.optim.lr_scheduler): Scheduler used to adjust the learning rate.
    - training (bool): Flag to indicate whether the model is being trained or evaluated.

    Returns:
    - A tuple containing the average loss for the epoch and the current learning rate (in training mode)
      or a tuple containing the evaluation loss and the model's predictions (in evaluation mode).
    """
    # Initialize the loss function; use 'mean' reduction in training and 'none' for evaluation
    loss_metric = nn.MSELoss(reduction='mean' if training else 'none')
    feature_dim = dataO.shape[1]  # Determine the feature dimension from the target outputs

    # Check if the model is a 'TransNAS_TSAD' model for custom processing
    if 'TransNAS_TSAD' in model.name:
        # For 'TransNAS_TSAD', use 'none' reduction for the loss to handle custom loss calculation
        loss_metric = nn.MSELoss(reduction='none')
        # Convert the input data to a tensor and create a dataset for DataLoader
        data_tensor = torch.DoubleTensor(data)
        data_set = TensorDataset(data_tensor, data_tensor)
        # Determine batch size based on training mode
        batch_size = model.batch if training else len(data)
        data_loader = DataLoader(data_set, batch_size=batch_size)

        losses = []  # List to store individual loss values for averaging
        cycle = epoch + 1  # Adjust cycle count for use in loss calculation

        if training:
            # Training mode: iterate over batches and perform optimization
            for batch_data, _ in data_loader:
                local_batch_size = batch_data.shape[0]
                # Rearrange data for processing by model
                sequence = batch_data.permute(1, 0, 2)
                target_element = sequence[-1, :, :].view(1, local_batch_size, feature_dim)
                # Generate model output
                model_output = model(sequence, target_element)
                # Calculate loss; handle differently if model_output is a tuple
                loss_val = loss_metric(model_output, target_element) if not isinstance(model_output, tuple) else (1 / cycle) * loss_metric(model_output[0], target_element) + (1 - 1/cycle) * loss_metric(model_output[1], target_element)
                if isinstance(model_output, tuple): model_output = model_output[1]  # Use the second element if output is a tuple
                losses.append(torch.mean(loss_val).item())  # Store the average loss
                total_loss = torch.mean(loss_val)
                optimizer.zero_grad()
                total_loss.backward(retain_graph=True)  # Backpropagate
                optimizer.step()  # Update model parameters
            scheduler.step()  # Update learning rate
            return np.mean(losses), optimizer.param_groups[0]['lr']  # Return average loss and learning rate
        else:
            # Evaluation mode: calculate loss without updating model parameters
            for batch_data, _ in data_loader:
                sequence = batch_data.permute(1, 0, 2)
                target_element = sequence[-1, :, :].view(1, batch_size, feature_dim)
                model_output = model(sequence, target_element)
                if isinstance(model_output, tuple): model_output = model_output[1]
            eval_loss = loss_metric(model_output, target_element)[0]  # Calculate evaluation loss
            return eval_loss.detach().numpy(), model_output.detach().numpy()[0]  # Return loss and predictions
