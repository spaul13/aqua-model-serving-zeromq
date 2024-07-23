from __future__ import division
import random
import numpy as np
import torch
import os, sys
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as tvt
import torch.optim as optim
#from torchsummary import summary           
import numpy as np
import re
import math
import copy
import time
import torchvision.models as models
from PIL import Image
import cv2

device = torch.device("cpu")#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
image_size = [299, 299]
transform = tvt.Compose([tvt.Resize(299),tvt.CenterCrop(299),tvt.ToTensor(),tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]) #for inception

#inputs
path_saved_model = sys.argv[1]
imgName = sys.argv[2]

class InceptionA(nn.Module):

	def __init__(self, in_channels, pool_features):
		super(InceptionA, self).__init__()
		self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

		self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
		self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

		self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
		self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
		self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

		self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

	def forward(self, x):
		branch1x1 = self.branch1x1(x)

		branch5x5 = self.branch5x5_1(x)
		branch5x5 = self.branch5x5_2(branch5x5)

		branch3x3dbl = self.branch3x3dbl_1(x)
		branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
		branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

		branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
		branch_pool = self.branch_pool(branch_pool)

		outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
		return torch.cat(outputs, 1)

class BasicConv2d(nn.Module):

	def __init__(self, in_channels, out_channels, **kwargs):
		super(BasicConv2d, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
		self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		return F.relu(x, inplace=True)

class Detection_net(nn.Module):
	def __init__(self):
		super(Detection_net, self).__init__()


	class Net(nn.Module):

		def __init__(self):
			super(Detection_net.Net, self).__init__()
			self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
			self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
			self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
			self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
			self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
			self.Mixed_5b = InceptionA(192, pool_features=32)
			self.fc_seqn = nn.Sequential(
				nn.Linear(12544, 1000),#12544
				nn.ReLU(inplace=True),
				nn.Linear(1000,100)
			)

		def forward(self, x):
			# 299 x 299 x 3
			x = self.Conv2d_1a_3x3(x)
			# 149 x 149 x 32
			x = self.Conv2d_2a_3x3(x)
			# 147 x 147 x 32
			x = self.Conv2d_2b_3x3(x)
			# 147 x 147 x 64
			x = F.max_pool2d(x, kernel_size=3, stride=2)
			# 73 x 73 x 64
			x = self.Conv2d_3b_1x1(x)
			# 73 x 73 x 80
			x = self.Conv2d_4a_3x3(x)
			# 71 x 71 x 192
			x = F.max_pool2d(x, kernel_size=3, stride=2)
			# 35 x 35 x 192
			x = self.Mixed_5b(x)
			# 35 x 35 x 256
			x = F.max_pool2d(x, kernel_size=5, stride=5)
			x = x.view(x.size(0), -1)
			x = self.fc_seqn(x)

			return x





	def run_code_for_testing(self, net, filename):
		net.load_state_dict(torch.load(path_saved_model,map_location=torch.device('cpu')))
		net.eval().to(device)
		#cv2 then PIL
		#im = cv2.imread(filename)
		#im1 = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
		#images = Image.fromarray(im1)
		
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
