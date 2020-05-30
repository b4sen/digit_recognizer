import torch
import torch.nn as nn
import torch.nn.functional as F
import typing
import numpy as np


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, 3)
		self.conv2 = nn.Conv2d(32, 32, 3)
		self.conv3 = nn.Conv2d(32, 64, 3)
		self.conv4 = nn.Conv2d(64, 128, 3)
		self.pool = nn.MaxPool2d(2,2)
		self.fc1 = nn.Linear(128*4*4, 128)
		self.fc2 = nn.Linear(128, 64)
		self.fc3 = nn.Linear(64, 10)

	def forward(self, x):
		x = self.conv1(x)
		x = F.relu(self.conv2(x))
		x = self.pool(x)
		x = self.conv3(x)
		x = F.relu(self.conv4(x))
		x = self.pool(x)
		#print(x.shape)
		x = x.view(-1, 128*4*4)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x


class CnnClassifier:

	def __init__(self, net: nn.Module, lr: float, wd: float):

		self.net = net
		if next(net.parameters()).is_cuda:
			self.device = "cuda"
		else:
			self.device = "cpu"
		self.net = self.net.to(self.device)
		self.loss = nn.CrossEntropyLoss()
		self.optim = torch.optim.SGD(self.net.parameters(), lr=lr, weight_decay=wd, momentum=0.9, nesterov=True)

	def train(self, data: np.ndarray, labels: np.ndarray) -> float:
		data_t = torch.tensor(data).unsqueeze(1).to(self.device)
		label_t = torch.LongTensor(labels).to(self.device)
		self.net.train()
		self.optim.zero_grad()

		loss = self.loss(self.net(data_t), label_t)
		loss.backward()
		self.optim.step()

		return loss.item()

	def predict(self, data: np.ndarray) -> np.ndarray:
		self.net.eval()
		data_t = torch.tensor(data).to(self.device)
		softmax = torch.nn.Softmax(dim=1)
		pred = softmax(self.net(data_t))
		
		return pred.detach().numpy()