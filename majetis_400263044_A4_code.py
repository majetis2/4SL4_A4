import tqdm
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

## Dataset Preparation Functions

def load_dataset(file_name):
    # Function to load the dataset
    raw_df = pd.read_csv(file_name, delimiter=',', header = None)
    X = raw_df.iloc[:, :4]
    t = raw_df.iloc[:, 4]
    return X, t

def split_dataset(random_seed, split, X, t):
    # Function to split the dataset in training, testing, and validation
    X_train, X_temp, t_train, t_temp = train_test_split(X, t, test_size = split, random_state = random_seed)
    X_test, X_val, t_test, t_val = train_test_split(X_temp, t_temp, test_size = 0.5, random_state = random_seed)
    return X_train, X_test, X_val, t_train, t_test, t_val

def standardize_features(X_train, X_test, X_val):
    # Function to standardize the training, testing, and validation dataset
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    X_val = sc.transform(X_val)
    return X_train, X_test, X_val

## Forward Propogation Functions

def sigmoid(x):
    # sigmoid activation function for thr output layer
    y = 1 /(1 + np.exp(-x))
    return y

def relu(x):
    # ReLU activation function for the hidden layers
    y = np.maximum(0, x)
    return y

def compute_layer_activation(output, activation_function):
    # Compute the h or layer activation output based on the activation function
    activation_output = (relu(output)).T if activation_function == 'relu' else (sigmoid(output)).T if activation_function == 'sigmoid' else -1
    return activation_output

def compute_layer_output(layer_features, weights_layer):
    # Compute the z or layer output from layer features and layer weights
    layer_output = weights_layer @ layer_features.T
    return layer_output

def get_layer(layer_input, weights, activation_function):
    # Compute the z or layer output and h or layer activation output based on layer features and weights
    ones = (np.ones(layer_input.shape[0])).reshape(-1, 1)
    prev_layer_features = np.hstack((ones, layer_input))
    output = compute_layer_output(prev_layer_features, weights)
    activation = compute_layer_activation(output, activation_function)
    return output, activation

def get_model(X, weights):
    # Compute z and h for all layers
    z_1, h_1 = get_layer(X, weights[0], 'relu')
    z_2, h_2 = get_layer(h_1, weights[1],'relu')
    z_3, h_3 = get_layer(h_2, weights[2], 'sigmoid')

    z = [z_1, z_2, z_3]
    h = [h_1, h_2, h_3]

    return z, h

## Backward Propogation Functions

def derivative_sigmoid(t, x):
    # Compute the derivative of sigmoid function
    y = sigmoid(x) - t
    return y

def derivative_relu(x):
    # Compute the derivative of ReLU function
    y = np.where(x > 0, 1.0, 0.0)
    return y

def compute_cost_gradient_layer(next_layer_weights, next_layer_gradient, current_layer_output, activation_function, t_train = 0):
    # Comput the cost gradient of a layer
    cost_gradient_layer = (derivative_relu(current_layer_output)) if activation_function == 'relu' else (derivative_sigmoid(t_train, current_layer_output)) if activation_function == 'sigmoid' else -1

    if isinstance(next_layer_weights, np.ndarray):
        cost_gradient_layer = cost_gradient_layer * ((next_layer_weights[1:, :]) @ next_layer_gradient)

    return cost_gradient_layer

def compute_weight_gradient_layer(prev_layer_output, gradient):
    # Compute the weight gradient of a layer
    ones = (np.ones(prev_layer_output.shape[0])).reshape(-1, 1)
    prev_layer_output_new = np.hstack((ones, prev_layer_output))

    weight_gradient_layer = gradient @ prev_layer_output_new

    return weight_gradient_layer

def update_layer(prev_layer_output, current_layer_output, activation_function, next_layer_weights = -1, next_layer_gradient = -1, t_train = 0):
    # Compute the cost gradient and weight gradient of a layer
    gradient_J = compute_cost_gradient_layer(next_layer_weights=next_layer_weights, next_layer_gradient=next_layer_gradient, current_layer_output=current_layer_output, t_train=t_train, activation_function=activation_function)
    gradient_W = compute_weight_gradient_layer(prev_layer_output=prev_layer_output, gradient=gradient_J)

    return gradient_J, gradient_W

def update_model(X_train, t_train, W, Z):
    # Compute the weight gradients for all layers
    W_1, W_2, W_3 = W
    z_1, z_2, z_3 = Z

    gradient_J3, gradient_W3 = update_layer(prev_layer_output = z_2.T, current_layer_output = z_3, t_train=t_train, activation_function = 'sigmoid')

    gradient_J2, gradient_W2 = update_layer(prev_layer_output = z_1.T, current_layer_output = z_2, activation_function = 'relu', next_layer_weights = W_3.T, next_layer_gradient = gradient_J3)

    gradient_J1, gradient_W1 = update_layer(prev_layer_output = X_train, current_layer_output = z_1, activation_function = 'relu', next_layer_weights = W_2.T, next_layer_gradient = gradient_J2)

    gradient_W = [gradient_W1, gradient_W2, gradient_W3]

    return gradient_W

## Gradient Descent Functions

def initialize_layer_weights(X_size, n1, n2):
    # Initialize layer weights using uniform distribution
    w_1 = np.random.uniform(low=-np.sqrt(6/(X_size + n1)), high=np.sqrt(6/(X_size + n1)), size=(n1, X_size))
    zeros = (np.zeros(w_1.shape[0])).reshape(-1, 1)
    w_1 = np.hstack((zeros, w_1))

    w_2 = np.random.uniform(low=-np.sqrt(6/(n1 + n2)), high=np.sqrt(6/(n1 + n2)), size=(n2, n1))
    zeros = (np.zeros(w_2.shape[0])).reshape(-1, 1)
    w_2 = np.hstack((zeros, w_2))

    w_3 = np.random.uniform(low=-np.sqrt(6/(n2 + 1)), high=np.sqrt(6/(n2 + 1)), size=(1, n2))
    zeros = (np.zeros(w_3.shape[0])).reshape(-1, 1)
    w_3 = np.hstack((zeros, w_3))

    weights = [w_1, w_2, w_3]

    return weights

def get_new_weights(w, gradient_w, learning_rate):
    # Update the weights of the layer
    w_new = [0, 0, 0]

    w_new[0] = w[0] - (learning_rate * gradient_w[0])
    w_new[1] = w[1] - (learning_rate * gradient_w[1])
    w_new[2] = w[2] - (learning_rate * gradient_w[2])

    return w_new

def compute_output(X_val, weights):
    # Compute the output of the neural network
    z_1, h_1 = get_layer(X_val, weights[0], 'relu')
    z_2, h_2 = get_layer(h_1, weights[1],'relu')
    z_3, h_3 = get_layer(h_2, weights[2], 'sigmoid')

    return z_3, h_3

def compute_cross_entropy_loss(z, t):
    # Funtion to compute the cross-entropy loss
    z = z.flatten()
    t = t.squeeze()

    ce_loss = t * (np.logaddexp(0, -z)) + (1 - t) * (np.logaddexp(0, z))

    return np.mean(ce_loss)

def plot_learning_curves_SGD(epochs, train_losses, validation_losses, n1, n2):
    # Function to plot learning curves for SGD
    plt.title(f'SGD Learning Curves for ({n1},{n2})')
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Loss')

    plt.plot(epochs, train_losses, label = 'Training')
    plt.plot(epochs, validation_losses, label = 'Validation')

    plt.legend(loc = "upper right")
    plt.show()

def compute_norm(gradient_ws):
    # Function to compute the norm of the gradient
    norm_layer = [np.linalg.norm(gradient_w) for gradient_w in gradient_ws]
    norm_combined = np.sqrt(np.sum(np.square(norm_layer)))

    return norm_combined

def SGD(X_train, X_val, t_train, t_val, w, n1, n2, learning_rate, epochs):
    # SGD Function
    epochs_graph = []
    training_losses = []
    validation_losses = []

    w_init = w
    weights = []

    flag = 0

    for epoch in tqdm(range(epochs)):

        permutation = np.random.permutation(X_train.shape[0])

        X_train_shuffled = X_train[permutation]
        t_train_shuffled = (np.array(t_train).reshape(-1, 1))[permutation]

        for iteration in range(X_train.shape[0]):

            X_train_iteration = (X_train_shuffled[iteration, :]).reshape(1, -1)
            t_train_iteration = (t_train_shuffled[iteration, :]).reshape(1, -1)

            z, h = get_model(X_train_iteration, w)

            gradient_w = update_model(X_train_iteration, t_train_iteration, w, z)
            w = get_new_weights(w, gradient_w, learning_rate)

        epochs_graph.append(epoch)

        z, h = get_model(X_train, w)
        z_val, h_val = get_model(X_val, w)

        train_loss = compute_cross_entropy_loss(z[2], t_train)
        validation_loss = compute_cross_entropy_loss(z_val[2], t_val)

        training_losses.append(train_loss)
        validation_losses.append(validation_loss)
        weights.append(w)

        norm = compute_norm(gradient_w)

        if(norm <= 1e-4 and flag == 0 and epoch > 40):
            flag = 1
            epoch_min = epoch

    epoch_min = np.argmin(validation_losses) if flag == 0 else epoch_min
    plot_learning_curves_SGD(epochs_graph, training_losses, validation_losses, n1, n2)

    print("Minimum Validation Loss Epoch:", epoch_min)
    print("Minimum Validation Loss:", validation_losses[epoch_min])

    return [w_init, training_losses[epoch_min], validation_losses[epoch_min], weights[epoch_min]]

## Classification Functions

def compute_classification_metrics(y, beta, t):
    # Function to compute classification metrics
    predicted = np.where(y >= beta, 1, 0)

    TP = np.sum((t == 1) & (predicted == 1))
    FP = np.sum((t == 0) & (predicted == 1))
    TN = np.sum((t == 0) & (predicted == 0))
    FN = np.sum((t == 1) & (predicted == 0))

    return TP, FP, TN, FN

def compute_bayes_classifier(y, t):
    # Function to compute classification using bayes classifier
    TP, FP, TN, FN = compute_classification_metrics(y, 0.5, t)

    print("FP:", FP)
    print("FN:", FN)

    misclassification_rate = (FP + FN)/(TP + FP + TN + FN)

    return misclassification_rate

## Model Configuration Functions

def run_model_configuration(X_train_standardized, t_train, X_val_standardized, t_val, w, n1, n2):
    # Function to run the model given a model configuration
    return SGD(np.array(X_train_standardized), np.array(X_val_standardized), np.array(t_train).reshape(1, -1), np.array(t_val).reshape(-1, 1), w, n1, n2, 0.001, 250)

## Main Function
def main():

    random_seed = 3044

    file_name = 'data_banknote_authentication.txt'

    ## Load the dataset from the text file
    X, t = load_dataset(file_name)

    ## Split the dataset into training, testing, and validation
    train_test_val_split = 0.20

    X_train, X_test, X_val, t_train, t_test, t_val = split_dataset(random_seed, train_test_val_split, X, t)

    ## Standardizing the data
    X_train_standardized, X_test_standardized, X_val_standardized = standardize_features(X_train, X_test, X_val)

    ## Initializing (n1, n2) configurations
    n1_n2_configurations = [(1, 1), (1, 2), (2, 2), (3, 1), (2, 4), (3, 3), (2, 5), (4, 3), (2, 6), (3, 5)]
    model_data_set = {key : [] for key in n1_n2_configurations}

    ## Getting data for all (n1, n2) configuration
    data = []

    ## Running (n1, n2) configuration for three different weights initialization and collecting the data
    for n1_n2_configuration in n1_n2_configurations:
        n1 = n1_n2_configuration[0]
        n2 = n1_n2_configuration[1]

        w_init_1 = initialize_layer_weights(4, n1, n2)
        w_init_2 = initialize_layer_weights(4, n1, n2)
        w_init_3 = initialize_layer_weights(4, n1, n2)

        current_row = [n1_n2_configuration]

        row_data_1 = current_row + run_model_configuration(X_train_standardized, t_train, X_val_standardized, t_val, w_init_1, n1, n2)
        row_data_2 = current_row + run_model_configuration(X_train_standardized, t_train, X_val_standardized, t_val, w_init_2, n1, n2)
        row_data_3 = current_row + run_model_configuration(X_train_standardized, t_train, X_val_standardized, t_val, w_init_3, n1, n2)

        data.append(row_data_1)
        data.append(row_data_2)
        data.append(row_data_3)

        row_data_combined = [row_data_1, row_data_2, row_data_3]
        val_loss = [row_data[3] for row_data in row_data_combined]
        min_val_loss_init = np.argmin(val_loss)
        min_weights = row_data_combined[min_val_loss_init][4]
        average_val_loss = np.mean(val_loss)

        model_data_set[n1_n2_configuration].append(average_val_loss)
        model_data_set[n1_n2_configuration].append(min_weights)

    ## Defining column names
    columns = ['n1_n2', 'W_init', 'Training Loss', 'Validation Loss', 'Weights']

    # Creating a dataframe df from data
    df = pd.DataFrame(data, columns=columns)

    ## Finding the best model configuration and best model weights using validation losses
    val_losses_final = [model_data_set[key][0] for key in model_data_set.keys()]
    best_model_index = np.argmin(val_losses_final)
    best_model_configuration = list(model_data_set.keys())[best_model_index]

    print("Best Model Configuration is:", best_model_configuration)

    best_model_weights = model_data_set[best_model_configuration][1]

    print("Best Model Weights is:", best_model_weights)

    ## Computing the output z and activation output h for training, testing, and validation using best model weights
    z_train, h_train = get_model(X_train_standardized, best_model_weights)
    z_test, h_test = get_model(X_test_standardized, best_model_weights)
    z_val, h_val = get_model(X_val_standardized, best_model_weights)

    ## Computing the misclassification rate on training, testing, and validation
    misclassification_train = compute_bayes_classifier(h_train[2], np.array(t_train).reshape(-1, 1))
    misclassification_test = compute_bayes_classifier(h_test[2], np.array(t_test).reshape(-1, 1))
    misclassification_val = compute_bayes_classifier(h_val[2], np.array(t_val).reshape(-1, 1))

    ## Printing the misclassification error on training, testing, and validation
    print("Misclassification Error on Training:", misclassification_train)
    print("Misclassification Error on Testing:", misclassification_test)
    print("Misclassification Error on Validation:", misclassification_val)

    return df

## Calling main function and getting the data into csv
data = main()

## Writing the data to a csv file
data.to_csv('reduced_initialization.csv', index=False)
