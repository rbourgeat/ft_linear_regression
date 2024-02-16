import csv
import numpy as np
import sys

def parsing(file_path):
    mileage = []
    price = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        if 'km' not in reader.fieldnames or 'price' not in reader.fieldnames:
            print("Error: 'km' or 'price' column not found in the CSV file.")
            sys.exit(1)
        for row in reader:
            try:
                mileage.append(float(row['km']))
                price.append(float(row['price']))
            except ValueError:
                print("Error: Non-numeric value found in 'km' or 'price' column.")
                sys.exit(1)
    if len(mileage) < 2 or len(price) < 2:
        print("Error: Insufficient data points. At least 2 values of 'km' and 'price' are required.")
        sys.exit(1)
    return mileage, price

def estimate_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)

def fit_linear_regression(mileage, price):
    X = np.array(mileage).reshape(-1, 1)
    y = np.array(price)
    X_b = np.c_[np.ones((len(X), 1)), X]
    theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return theta[0], theta[1]

def main(file_path='data.csv'):
    mileage, price = parsing(file_path)
    theta0, theta1 = fit_linear_regression(mileage, price)
    while True:
        try:
            input_mileage = float(input("Enter the mileage: "))
            if input_mileage < 0:
                raise ValueError("Mileage cannot be negative.")
            break
        except ValueError as ve:
            print("Invalid input: Please enter a valid mileage.")
    
    estimated_price = estimate_price(input_mileage, theta0, theta1)
    print(f"The estimated price for a car with {input_mileage} km mileage is ${estimated_price:.2f}.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
