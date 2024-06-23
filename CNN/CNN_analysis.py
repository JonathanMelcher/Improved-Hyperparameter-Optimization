import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
from numpy.linalg import norm


import sys

# Add the directory containing the modules to Python's search path
sys.path.append('../Grid_Generation')

# Now you can import your modules
import Make_grids as hg
import GeneratingMultidimensional as gm

import time
import argparse


from importlib import reload

reload(hg)



start_time = time.time()  # Record the start time


parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--res', type=int, default=3,
                    help='resolution of grid', )


# Move the model to the GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

train_dataset = datasets.MNIST('./data', train=True, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ]))


# Define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)  # 12*12*64 = 9216, adjust this based on your conv2d output
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

patience = 2


def train_model(lr, eps, w_d, gamma, train_loader, val_loader):
    # Define a new ResNet-50 model
    model = Net().to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=eps, weight_decay=w_d)

    # Initialize learning rate scheduler
    scheduler = StepLR(optimizer, step_size=5, gamma=gamma)

    num_epochs = 15
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Train the model
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch}, Loss: {running_loss / len(train_loader)}")

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
        average_val_loss = total_val_loss / len(val_loader)

        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            print(f'Early stopping triggered at epoch {epoch}')
            break

        # Update the learning rate
        scheduler.step()

    return running_loss / len(train_loader)



# Train a model for each set of hyperparameters
resolution = parser.parse_args().res



num_iterations = 20
num_variables = 4
num_samples = resolution**num_variables



lr_max = 0.01
lr_min = 0.0001
eps_max = 1e-5
eps_min = 1e-10
w_d_max = 0.5e-3
w_d_min = 0.5e-5
gamma_max = 0.5
gamma_min = 0.05

def make_grid_from_arrays(arrays):
	grid = np.array(np.meshgrid(*arrays)).T.reshape(-1, len(arrays))
	return grid


lr_grid = np.linspace(lr_min, lr_max, resolution)
eps_grid = np.linspace(eps_min, eps_max, resolution)
w_d_grid = np.linspace(w_d_min, w_d_max, resolution)
gamma_grid = np.linspace(gamma_min, gamma_max, resolution)

param_grid = make_grid_from_arrays([lr_grid, eps_grid, w_d_grid, gamma_grid])


print(f'Initializing with resolution {resolution} and num variables {num_variables}')
grid_search = []
random_search = []
RSA_search = []
gridrandom_search = []
tilling_search = []
for i in range(num_iterations):
	generator_seed = torch.Generator(device = 'cpu')
	generator_seed.manual_seed(i)
	train_indices, val_indices=train_test_split(np.arange(len(train_dataset)),test_size=0.2,random_state=i)
	train_subset = Subset(train_dataset, train_indices)
	val_subset = Subset(train_dataset, val_indices)
	
	subset_size = np.random.randint(12500, 15000, dtype = int)
	indices = torch.randint(0, len(train_subset), (subset_size,), dtype=torch.int64)
	sampler = SubsetRandomSampler(indices, generator = generator_seed)
	train_ite = torch.utils.data.DataLoader(train_subset, batch_size=1000, sampler=sampler, generator = generator_seed)
	val_ite = torch.utils.data.DataLoader(val_subset, batch_size=1000, shuffle = False)
	print(f'Running iteration number {int(i+1)}, with data size {len(train_ite)}')


	loss_grid_appended = []
	for param_x, param_y, param_w, param_z in zip(param_grid[:,0], param_grid[:,1], param_grid[:,2], param_grid[:,3]):
		loss_grid = train_model(param_x, param_y, param_w, param_z, train_ite, val_ite)
		loss_grid_appended.append([param_x, param_y, param_w, param_z, loss_grid])
		#print(f'Grid search iteration {i+1} of {num_samples} complete')
		#i += 1

	grid_search.append(loss_grid_appended)

	learning_rates_random = np.random.uniform(lr_min, lr_max, num_samples)
	epsilon_random = np.random.uniform(eps_min, eps_max, num_samples)
	w_d_random = np.random.uniform(w_d_min, w_d_max, num_samples)
	gamma_random = np.random.uniform(gamma_min, gamma_max, num_samples)

	loss_random_appended = []
	for param_x, param_y, param_w, param_z in zip(learning_rates_random, epsilon_random, w_d_random, gamma_random):
		loss_random = train_model(param_x, param_y, param_w, param_z, train_ite, val_ite)
		loss_random_appended.append([param_x, param_y, param_w, param_z, loss_random])
		#print(f'Grid search iteration {i+1} of {num_samples} complete')
		#i += 1

	random_search.append(loss_random_appended)

	#print('Making RSA')
	RSA = hg.make_RSA(num_samples, 1, num_variables)
	RSA = np.array(RSA)

	learning_rates_RSA = hg.transform_grid(RSA[:,0], lr_min, lr_max)
	epsilon_RSA = hg.transform_grid(RSA[:,1], eps_min, eps_max)
	w_d_RSA = hg.transform_grid(RSA[:,2], w_d_min, w_d_max)
	gamma_RSA = hg.transform_grid(RSA[:,3], gamma_min, gamma_max)


	loss_RSA_appended = []


	# RSA
	#print('Running RSA')
	for param_x, param_y, param_w, param_z in zip(learning_rates_RSA, epsilon_RSA, w_d_RSA, gamma_RSA):
		loss_RSA = train_model(param_x, param_y, param_w, param_z, train_ite, val_ite)
		loss_RSA_appended.append([param_x, param_y, param_w, param_z, loss_RSA])
		#print(f'RSA iteration {i+1} of {num_samples} complete')
		#i += 1
    

	RSA_search.append(loss_RSA_appended)

	#print('Making gridrandom')
	gridrandom = hg.make_gridrandom(num_samples, 1, num_variables)
	gridrandom = np.array(gridrandom)

	learning_rates_gridrandom = hg.transform_grid(gridrandom[:,0], lr_min, lr_max)
	epsilon_gridrandom = hg.transform_grid(gridrandom[:,1], eps_min, eps_max)
	w_d_gridrandom = hg.transform_grid(gridrandom[:,2], w_d_min, w_d_max)
	gamma_gridrandom = hg.transform_grid(gridrandom[:,3], gamma_min, gamma_max)


	loss_gridrandom_appended = []


	for param_x, param_y, param_w, param_z in zip(learning_rates_gridrandom, epsilon_gridrandom, w_d_gridrandom, gamma_gridrandom):
		loss_gridrandom = train_model(param_x, param_y, param_w, param_z, train_ite, val_ite)
		loss_gridrandom_appended.append([param_x, param_y, param_w, param_z, loss_gridrandom])

	gridrandom_search.append(loss_gridrandom_appended)


	#print('Making gridrandom')
	tilling = hg.make_true_hyper(num_samples, 1, num_variables)
	tilling = np.array(tilling)

	learning_rates_tilling = hg.transform_grid(tilling[:,0], lr_min, lr_max)
	epsilon_tilling = hg.transform_grid(tilling[:,1], eps_min, eps_max)
	w_d_tilling = hg.transform_grid(tilling[:,2], w_d_min, w_d_max)
	gamma_tilling = hg.transform_grid(tilling[:,3], gamma_min, gamma_max)


	loss_tilling_appended = []

	# RSA
	#print('Running RSA')
	for param_x, param_y, param_w, param_z in zip(learning_rates_tilling, epsilon_tilling, w_d_tilling, gamma_tilling):
		loss_tilling = train_model(param_x, param_y, param_w, param_z, train_ite, val_ite)
		loss_tilling_appended.append([param_x, param_y, param_w, param_z, loss_tilling])
		#print(f'RSA iteration {i+1} of {num_samples} complete')
		#i += 1
    

	tilling_search.append(loss_tilling_appended)

pickle.dump(RSA_search, open('values_RSA_d_{}_res{}.p'.format(num_variables, resolution), 'wb'))
pickle.dump(random_search, open('values_random_d_{}_res{}.p'.format(num_variables, resolution), 'wb'))
pickle.dump(grid_search, open('values_grid_d_{}_res{}.p'.format(num_variables, resolution), 'wb'))
pickle.dump(gridrandom_search, open('values_gridrandom_d_{}_res{}.p'.format(num_variables, resolution), 'wb'))
pickle.dump(tilling_search, open('values_tilling_d_{}_res{}.p'.format(num_variables, resolution), 'wb'))



end_time = time.time()    # Record the end time
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")