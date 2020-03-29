import os
import numpy as np 

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

EPSILON = 1e-10

def marginalPdf(values, bins, sigma):

	residuals = values - bins.unsqueeze(0).unsqueeze(0)
	kernel_values = torch.exp(-0.5*(residuals / sigma).pow(2))
	
	pdf = torch.mean(kernel_values, dim=1)
	normalization = torch.sum(pdf, dim=1).unsqueeze(1) + EPSILON
	pdf = pdf / normalization

	return pdf, kernel_values


def jointPdf(kernel_values1, kernel_values2):

	joint_kernel_values = torch.matmul(kernel_values1.transpose(1, 2), kernel_values2) 
	normalization = torch.sum(joint_kernel_values, dim=(1,2)).view(-1, 1, 1) + EPSILON
	pdf = joint_kernel_values / normalization

	return pdf


def histogram(x, bins, bandwidth):
	'''
		x: tensor of shape BxN
		bins: tensor of length num_bins
		bandwidth: gaussian smoothing factor

		return: normalized histogram of x
	'''
	x = x*255
	pdf, _ = marginalPdf(x.unsqueeze(2), bins, bandwidth)

	return pdf


def histogram2d(x1, x2, bins, bandwidth):
	'''
		values: tensor of shape BxN
		bins: tensor of length num_bins
		bandwidth: gaussian smoothing factor
	'''
	x1 = x1*255
	x2 = x2*255

	pdf1, kernel_values1 = marginalPdf(x1.unsqueeze(2), bins, bandwidth)
	pdf2, kernel_values2 = marginalPdf(x2.unsqueeze(2), bins, bandwidth)

	joint_pdf = jointPdf(kernel_values1, kernel_values2)
	
	return joint_pdf


if __name__ == '__main__':
	
	device = 'cuda:0'

	### Create test cases ###
	img1 = Image.open('grad1.jpg').convert('L')
	img2 = Image.open('grad.jpg').convert('L')

	arr1 = np.array(img1)
	arr2 = np.array(img2)

	img1 = transforms.ToTensor() (img1).unsqueeze(dim=0).to(device)
	img2 = transforms.ToTensor() (img2).unsqueeze(dim=0).to(device)

	# Pair of different images, pair of same images
	input1 = torch.cat([img2, img2])
	input2 = torch.cat([img1, img2])

	B, C, H, W = input1.shape

	joint_pdf = histogram2d(input1.view(B, H*W), input2.view(B, H*W), torch.linspace(0,255,256).to(device), 2*0.4**2)
	plt.imshow(joint_pdf[0].cpu().numpy())
	plt.colorbar()
	plt.show()

	pdf = histogram(input1.view(B, H*W), torch.linspace(0,255,256).to(device), 2*0.7**2)
	plt.plot(np.linspace(0,255,256), pdf[0].cpu().numpy())
	plt.hist(arr2.ravel(), np.linspace(0,255,256), density=True)
	plt.show()