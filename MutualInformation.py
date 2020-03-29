import os
import numpy as np 

import torch
import torch.nn as nn

import skimage.io
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

from sklearn.metrics import normalized_mutual_info_score


class MutualInformation(nn.Module):

	def __init__(self, sigma=0.4, num_bins=256, normalize=True, device='cpu'):
		super(MutualInformation, self).__init__()

		self.sigma = 2*sigma**2
		self.num_bins = num_bins
		self.normalize = normalize

		self.bins = torch.linspace(0, 255, num_bins, device=device).float()


	def marginalPdf(self, values):

		residuals = values - self.bins.unsqueeze(0).unsqueeze(0)
		kernel_values = torch.exp(-0.5*(residuals / self.sigma).pow(2))
		
		pdf = torch.mean(kernel_values, dim=1)
		normalization = torch.sum(pdf, dim=1).unsqueeze(1) + 1e-8
		pdf = pdf / normalization
		
		return pdf, kernel_values


	def jointPdf(self, kernel_values1, kernel_values2):

		joint_kernel_values = torch.matmul(kernel_values1.transpose(1, 2), kernel_values2) 
		normalization = torch.sum(joint_kernel_values, dim=(1,2)).view(-1, 1, 1) + 1e-8
		pdf = joint_kernel_values / normalization

		return pdf


	def getMutualInformation(self, input1, input2):
		'''
			input1: B, C, H, W
			input2: B, C, H, W

		'''

		# Torch tensors for images between (0, 1)
		input1 = input1*255
		input2 = input2*255

		B, C, H, W = input1.shape
		assert((input1.shape == input2.shape))

		x1 = input1.view(B, H*W, C)
		x2 = input2.view(B, H*W, C)
		
		pdf_x1, kernel_values1 = self.marginalPdf(x1)
		pdf_x2, kernel_values2 = self.marginalPdf(x2)
		pdf_x1x2 = self.jointPdf(kernel_values1, kernel_values2)

		H_x1 = -torch.sum(pdf_x1*torch.log2(pdf_x1 + 1e-10), dim=1)
		H_x2 = -torch.sum(pdf_x2*torch.log2(pdf_x2 + 1e-10), dim=1)
		H_x1x2 = -torch.sum(pdf_x1x2*torch.log2(pdf_x1x2 + 1e-10), dim=(1,2))

		mutual_information = H_x1 + H_x2 - H_x1x2
		
		if self.normalize:
			mutual_information = 2*mutual_information/(H_x1+H_x2)

		return mutual_information


	def forward(self, input1, input2):
		return self.getMutualInformation(input1, input2)



if __name__ == '__main__':
	
	device = 'cuda:0'

	### Create test cases ###
	img1 = Image.open('grad.jpg').convert('L')
	img2 = img1.rotate(10)

	arr1 = np.array(img1)
	arr2 = np.array(img2)
	
	mi_true_1 = normalized_mutual_info_score(arr1.ravel(), arr2.ravel())
	mi_true_2 = normalized_mutual_info_score(arr2.ravel(), arr2.ravel())

	img1 = transforms.ToTensor() (img1).unsqueeze(dim=0).to(device)
	img2 = transforms.ToTensor() (img2).unsqueeze(dim=0).to(device)

	# Pair of different images, pair of same images
	input1 = torch.cat([img1, img2])
	input2 = torch.cat([img2, img2])

	MI = MutualInformation(device='cuda:0', num_bins=256, sigma=0.4, normalize=True)
	mi_test = MI(input1, input2)

	mi_test_1 = mi_test[0].cpu().numpy()
	mi_test_2 = mi_test[1].cpu().numpy()

	print('Image Pair 1 | sklearn MI: {}, this MI: {}'.format(mi_true_1, mi_test_1))
	print('Image Pair 2 | sklearn MI: {}, this MI: {}'.format(mi_true_2, mi_test_2))

	assert(np.abs(mi_test_1 - mi_true_1) < 0.05)
	assert(np.abs(mi_test_2 - mi_true_2) < 0.05)