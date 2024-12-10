# Naive bayes classifier model for MNIST dataset training 

import matplotlib.pyplot as plt
import numpy as np 

# Bernoulli naive bayes
class BNB:
    def __init__(self):
        self.priors = None  # P(C_k)
        self.likelihoods = None  # P(x_i | C_k)
        self.classes = None  # Unique class labels

    def fit(self, x_train, y_train):
        self.classes = np.unique(y_train)  # Unique classes
        n_classes = len(self.classes)
        n_features = x_train.shape[1]

        # Estimate priors P(C_k)
        self.priors = np.zeros(n_classes)
        for idx, c in enumerate(self.classes):
            #self.priors[idx] = np.sum(y_train == c) / len(y_train)  # Relative frequency
            self.priors[idx] = 1 / n_classes

        # Estimate likelihoods P(x_i | C_k) with Laplace smoothing
        self.likelihoods = np.zeros((n_classes, n_features))
        for idx, c in enumerate(self.classes):
            X_class = x_train[y_train == c]
            self.likelihoods[idx] = (np.sum(X_class, axis=0) + 0.5) / (X_class.shape[0] + 1)
        #print(f"shape of likelihoods before prediction: {self.likelihoods.shape}")
        return self.priors, self.likelihoods

    def predict(self, X):
        #print(f"shape of image: {X.shape}")
        # Matrix computation of log-probabilities
        log_priors = np.log(self.priors)
        # computing the likelihood P(I | C_k) for all classes C_k
        log_likelihoods = (X @ np.log(self.likelihoods.T)) + ((1 - X) @ np.log(1 - self.likelihoods.T))
        # log(P(C_k | I)) = log(P(I | C_k) + log(P(C_k)))
        log_probs = log_priors + log_likelihoods  # Broadcasting for each class

        # Return class with highest log-probability
        return self.classes[np.argmax(log_probs, axis=1)]

# downsamples a non-square array to 28x28 resolution by using block-based averaging.
def downsample_to_28x28_nonsquare(array):
    orig_height, orig_width = array.shape

    # Calculate block sizes
    block_height = orig_height / 28
    block_width = orig_width / 28

    # Initialize the downsampled array
    downscaled = np.zeros((28, 28), dtype=np.uint8)

    for i in range(28):
        for j in range(28):
            # Calculate the block boundaries
            start_row = int(i * block_height)
            end_row = int((i + 1) * block_height)
            start_col = int(j * block_width)
            end_col = int((j + 1) * block_width)

            # Extract the block
            block = array[start_row:end_row, start_col:end_col]

            # Downscale by summing block values and thresholding
            downscaled[i, j] = 1 if np.sum(block) > 0 else 0

    return downscaled