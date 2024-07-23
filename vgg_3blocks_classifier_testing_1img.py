from __future__ import division
import random, timeit
import numpy as np
import torch
import os, sys
import torch.nn as nn
import torch.nn.functional as F
import torchvision                  
import torchvision.transforms as tvt
import torch.optim as optim
import numpy as np
import re
import math
import copy
import time
from sklearn.metrics import mean_squared_error
import scipy.stats as stats
from scipy.stats import pearsonr
import torchvision.models as models
import cv2
from PIL import Image

label_list, predicted_list = [], []

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")#cuda:0" if torch.cuda.is_available() else "cpu")

transform = tvt.Compose([tvt.Resize(256),tvt.CenterCrop(224),tvt.ToTensor(),tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
#inputs
path_saved_model = sys.argv[1]
imgName = sys.argv[2]



class Detection_net(nn.Module):
	def __init__(self):
		super(Detection_net, self).__init__()
	class Net(nn.Module):
		def __init__(self):
			super(Detection_net.Net, self).__init__()
			self.conv_seqn = nn.Sequential(
				# Conv Layer block 1:
				nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
				nn.ReLU(inplace=True),
				nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
				nn.ReLU(inplace=True),
				nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
				nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
				nn.ReLU(inplace=True),
				nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
				nn.ReLU(inplace=True),
				nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
				nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
				nn.ReLU(inplace=True),
				nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
				nn.ReLU(inplace=True),
				nn.MaxPool2d(kernel_size=8, stride=8, padding=0, dilation=1, ceil_mode=False),
			)
			self.max_pool = nn.MaxPool2d(kernel_size=8, stride=8)
			self.fc_seqn = nn.Sequential(
				#nn.ReLU(inplace=True),
				#nn.Dropout(p=0.1),
				nn.Linear(12544, 1000),
				nn.ReLU(inplace=True),
				nn.Linear(1000,100)
			)

		def forward(self, x):
			x = self.conv_seqn(x)
			# flatten
			#x = self.max_pool(x)
			x = x.view(x.size(0), -1)
			x = self.fc_seqn(x)
			return x

	def save_model(self, model):
		torch.save(model.state_dict(), self.path_saved_model)

	def show_network_summary(self, model):
		summary(model, (3,image_size[0],image_size[1]),-1, device='cpu')
	
	def run_code_for_testing(self, net, filename):
		net.load_state_dict(torch.load(path_saved_model,map_location=torch.device('cpu')))
		net.eval().to(device)
		
		images = Image.open(filename)
		images = transform(images)
		images = torch.unsqueeze(images, 0)
		with torch.no_grad():
			outputs = net(images.to(device))
			_, predicted = torch.max(outputs.data, 1)
			soft_output = F.softmax(outputs.data, dim=1)
			print(filename, outputs[0][predicted.cpu().item()].cpu().item(), predicted.cpu().item(), soft_output.data[0][predicted.cpu().item()].cpu().item())


def main():
        detnet = Detection_net()
        net = detnet.Net()
        detnet.run_code_for_testing(net,imgName)

if __name__== "__main__":
        main()
				

