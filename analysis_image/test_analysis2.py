# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import sys
sys.path.append('./data')
from data import img2numpy_arr, generate_patches

import argparse
import numpy as np
import tflearn as tl
import tflearn.data_utils as du

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--t', action='store', dest='test_path', type=str, help='Test Data Path')
	config = parser.parse_args()

	#Load Test data
	image_count = (3,6)
	patch_count = 20
	X = generate_patches(img2numpy_arr(config.test_path), image_count, patch_count)

	# Building Residual Network
	net = tl.input_data(shape=[None, 42, 42, 3])
	net = tl.conv_2d(net, 32, 3)
	net = tl.batch_normalization(net)
	net = tl.activation(net, 'relu')
	net = tl.shallow_residual_block(net, 4, 32, regularizer='L2')
	net = tl.shallow_residual_block(net, 1, 32, downsample=True,
												 regularizer='L2')
	net = tl.shallow_residual_block(net, 4, 64, regularizer='L2')
	net = tl.shallow_residual_block(net, 1, 64, downsample=True,
												 regularizer='L2')
	net = tl.shallow_residual_block(net, 5, 64, regularizer='L2')
	net = tl.global_avg_pool(net)
	
	# Regression
	net = tl.fully_connected(net, 9, activation='softmax')
	mom = tl.Momentum(0.1, lr_decay=0.1, decay_step=16000, staircase=True)
	net = tl.regression(net, optimizer=mom,
									 loss='categorical_crossentropy')
	# Training
	model = tl.DNN(net)
	model.load('resnet_analysis-172500')
	pred_y = model.predict(X)
	print(pred_y)
