import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# SKLEARN IS BEING USED FOR COMPARISION PURPOSES ONLY...
from sklearn.linear_model import LinearRegression as lin


# IMPORTING AND PREPROCESSING THE DATA FILE...

def load_data(loc):
    data = pd.read_csv(loc)
    data = data.drop(['name', 'origin'], axis=1)
    data = data.replace('?', np.NaN)
    data = data.dropna()
    data = np.float64(data)
    print("The data was loaded successfully! ")
    print(data.shape)
    print(data.dtype)
    return data, data.shape


# SEPARATING TEST AND TRAINING DATA...
def split_data(data):
    train_data = data[:int(data.shape[0] * 0.8), :]
    test_data = data[int(data.shape[0] * 0.8):, :]
    train_size = train_data.shape
    test_size = test_data.shape
    print(train_size, test_size)
    return train_data, test_data, train_size, test_size


# CREATING FEATURE MATRICES AND TARGET VECTORS...
def create_features(train_data, test_data):
    train_targets = train_data[:, 0]
    train_features = train_data[:, 1:]
    test_targets = test_data[:, 0]
    test_features = test_data[:, 1:]
    print("\n\n")
    return train_features, train_targets, test_features, test_targets


# NORMALIZE FEATURES...

def normalize_features(train_features, test_features):
    print("Features have been normalized, the max and min Z-scores are: \n")
    print("Training data: ")
    train_size = train_data.shape
    test_size = test_data.shape
    for i in range(train_size[1]):
        mu = np.mean(train_data[:, i])
        sigma = np.std(train_data[:, i])
        train_data[:, i] = (train_data[:, i] - mu) / sigma
        print(i, " : ", max(train_data[:, i]), " and ", min(train_data[:, i]))

    print("\nTesting data: ")
    for i in range(test_size[1]):
        mu = np.mean(test_data[:, i])
        sigma = np.std(test_data[:, i])
        test_data[:, i] = (test_data[:, i] - mu) / sigma
        print(i, " : ", max(test_data[:, i]), " and ", min(test_data[:, i]))
    print("\n\n")
    return train_features, test_features


# ADDING x0 FEATURES:


def add_bias(train_features, test_features):
    train_size = train_data.shape
    test_size = test_data.shape
    x0 = np.ones([train_size[0], 1])
    train_features = np.hstack([x0, train_features])
    tx0 = np.ones([test_size[0], 1])
    test_features = np.hstack([tx0, test_features])
    return train_features, test_features


# Defining cost function...

def cost(features, targets, params):
    m = targets.shape[0]
    j = np.sum((features.dot(params) - targets) ** 2) / (2 * m)
    return j


# Defining gradient descent algorithm...

def grad_des(features, targets, params, alpha, maxiter):
    history = [0] * maxiter
    m = targets.shape[0]
    for i in range(maxiter):
        pred = features.dot(params)
        delta = features.T.dot(pred - targets) / m
        params = params - alpha * delta
        j = cost(features, targets, params)
        history[i] = j
    return params, history


# Lets get this started...

data, data_size = load_data("mpg.csv")
train_data, test_data, train_size, test_size = split_data(data)
train_features, train_targets, test_features, test_targets = create_features(train_data, test_data)
train_features, test_features = normalize_features(train_features, test_features)
train_features, test_features = add_bias(train_features,test_features)

# Defining initial parameters...

ini_params = np.array([0] * train_features.shape[1])
alpha = 0.03
max_iter = 800

# Calling Gradient descent algorithm...

new_params, cost_history = grad_des(train_features, train_targets, ini_params, alpha, max_iter)

# Plots to help visualize effectiveness of learning rate...

plt.plot(range(1, max_iter + 1), cost_history)
plt.xlabel("ITERATIONS")
plt.ylabel("COST")
plt.title("Cost V/S Iterations")
plt.show()


print("The optimum parameters for the given data set is: \n", new_params)



# Error analysis and comparision to sklearn...

def root_mean_sq_err(targets, pred):
    rmse = np.sqrt(sum((targets - pred) ** 2) / targets.shape[0])
    return rmse


def r2_score(Y, Y_pred):
    mean_y = np.mean(Y)
    ss_tot = sum((Y - mean_y) ** 2)
    ss_res = sum((Y - Y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


# Printing Error values of my model...

print("\n\nError values: (Lower is better) : \nMY PREDICTOR:")
predictions = test_features.dot(new_params)
print("The root mean square error is: ", root_mean_sq_err(test_targets, predictions))
print("The r^2 score is: ", r2_score(test_targets, predictions), "\n\n")


# Building a linear regressor from Sklearn for comparision...

reg = lin()
reg.fit(train_features, train_targets)
skpred = reg.predict(test_features)

# Printing Sklearn's error values...

print("SKLEARN :")
print("The root mean square error of sklearn is: ", root_mean_sq_err(test_targets, skpred))
print("The r^2 score of sklearn is: ", r2_score(test_targets, skpred), "\n\n")


# Printing first 10 predictions of my model and sklearn side by side to actual values...

print("ACTUAL \t\t\t\t MY MODEL \t\t\t\t\t SKLEARN\n")
for i in range(10):
    print(test_targets[i], "\t", predictions[i], "\t", skpred[i])
