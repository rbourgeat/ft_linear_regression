# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    train.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: rbourgea <rbourgea@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/02/16 16:18:11 by rbourgea          #+#    #+#              #
#    Updated: 2024/02/19 14:49:15 by rbourgea         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple

def predict(theta0, theta1, mileage):
	return theta0 + (theta1 * mileage)

# Gradient descent algorithm
def train(mileages, targets, theta0, theta1, learning_rate):
	predictions = predict(theta0, theta1, mileages)
	errors = error(predictions, targets)
	m = errors.shape[0]
	delta0 = learning_rate * (1 / m) * np.sum(errors)
	delta1 = learning_rate * (1 / m) * np.sum(errors * mileages)
	return (theta0 - delta0, theta1 - delta1)

def normalize(data):
	max_km = data["km"].max()
	max_price = data["price"].max()
	data["km"] = data["km"] / max_km
	data["price"] = data["price"] / max_price
	return (max_km, max_price)

def save_weights(theta0, theta1):
	np.save("weights", np.array([theta0, theta1]))
	print("Training done: Weigths saved in weights.npy file !")

def error(prediction, target):
	return prediction - target

# MAE (mean absolute error) measures the average differences between predicted values ​​and actual values.
def get_averrage_error(mileages, targets, theta0, theta1):
	predictions = predict(theta0, theta1, mileages)
	errors = np.abs(error(predictions, targets))
	m = errors.shape[0]
	return (1 / m) * np.sum(errors)

def main():
	data = pd.read_csv("./data.csv")
	max_km, max_price = normalize(data)
	data = data.values
	mileages = data[:,0]
	targets = data[:,1]
	learning_rate = 0.1
	iterations = 1000
	batch_size = data.shape[0]
	errors = []
	theta0, theta1 = (0.0, 0.0)

	for iteration in range(1, iterations + 1):
		for b in range(0, data.shape[0], batch_size):
			theta0, theta1 = train(mileages[b:b + batch_size], targets[b:b + batch_size], theta0, theta1, learning_rate)
		avg_error = get_averrage_error(data[:, 0], data[:, 1], theta0, theta1)
		errors.append(avg_error)
		print("Iteration {:4}/{:4}, average error rate: {:.6f}".format(iteration, iterations, avg_error))

	fig, axs = plt.subplots(1, 2, figsize=(12, 5))

	axs[0].plot(np.array(errors))
	axs[0].set_xlabel('Iterations')
	axs[0].set_ylabel('Average Error')
	axs[0].legend(['Average Error'])
	axs[0].set_title('Average Error Over Iterations')

	axs[1].scatter(mileages * max_km, targets * max_price, label='Data Points')
	axs[1].set_xlabel('Mileage (km)')
	axs[1].set_ylabel('Price')
	axs[1].set_title('Linear Regression')
	axs[1].plot(mileages * max_km, predict(theta0, theta1, mileages) * max_price, color='red', label='Linear Regression')
	axs[1].legend()

	plt.tight_layout()
	plt.show()
	
	theta0 *= max_price
	theta1 *= (max_price / max_km)
	print("Theta0: {:.4f}".format(theta0))
	print("Theta1: {:.4f}".format(theta1))
	save_weights(theta0, theta1)

if (__name__ == "__main__"):
	main()
