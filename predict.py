# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    predict.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: rbourgea <rbourgea@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/02/16 16:18:18 by rbourgea          #+#    #+#              #
#    Updated: 2024/02/16 16:50:19 by rbourgea         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import csv
import numpy as np
from typing import Tuple

def load_model_weights():
	try:
		thetas = np.load("weights.npy")
		return (thetas[0], thetas[1])
	except:
		save_weights(0, 0)	
	return (0, 0)

def estimate_price(mileage, theta0, theta1):
	return theta0 + (theta1 * mileage)

def main():
	try:
		input_mileage = float(input("Enter the mileage: "))
	except ValueError as e:
		print("Invalid input: Please enter a valid mileage.")
		return

	theta0, theta1 = load_model_weights()

	estimated_price = estimate_price(input_mileage, theta0, theta1)
	print(f"The estimated price for a car with {input_mileage} km mileage is ${estimated_price:.2f}.")

if __name__ == "__main__":
	main()
