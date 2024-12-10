# displays all performance metrics for the naive bayes bernoulli classifier 
# out of sample error - 
# in sample error 

from naivebayes import BNB
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from naivebayes import downsample_to_28x28_nonsquare
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

test_data = pd.read_csv("mnist_test_final.csv")
train_data = pd.read_csv("mnist_train_final.csv")

x_test = test_data.drop(columns = ['label'])
y_test = test_data['label']
x_train = train_data.drop(columns = ['label'])
y_train = train_data['label']

# scale the data to be between 0 and 1 
x_test = x_test / 255
x_train = x_train / 255

# binarize the mnist data 
def binarize(data):
    # Use NumPy for element-wise operations, which is faster and more concise
    return (data > 0).astype(int)

x_train_binary = binarize(x_train)
x_test_binary = binarize(x_test)

array = np.load('canvas.npy')
cropped_array = np.load('bounding_box_canvas.npy')

# Method 1: Block-Based
downscaled_block = downsample_to_28x28_nonsquare(cropped_array)
#plt.imshow(downscaled_block, cmap='gray', vmin=0, vmax=1)
#plt.title('Downscaled with Block-Based Method')
#plt.show()


fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.suptitle('Binarization on MNIST data')

im3 = ax3.imshow(downscaled_block, cmap='gray', vmin=0, vmax=1, aspect='auto')
im2 = ax2.imshow(x_test_binary.iloc[0].values.reshape(28,28), cmap='gray', vmin=0, vmax=1, aspect='auto')
im1 = ax1.imshow(x_test.iloc[0].values.reshape(28,28), cmap='gray', vmin=0, vmax=1, aspect='auto')

ax1.set_title('Before binarization')
ax2.set_title('After binarization')
ax3.set_title('Sample image from paint program')

plt.tight_layout()
# Make space for title
plt.subplots_adjust(top=0.85)
plt.show()

#plt.imshow(x_test_binary.iloc[23].values.reshape(28,28), cmap='gray', vmin=0, vmax=1)
#plt.title(f"mnist number with true label {y_test.iloc[23]}")
#plt.show()

model = BNB()

# Fit the model
priors, likelihoods = model.fit(x_train_binary, y_train)

# Predict on the test set
y_pred = model.predict(x_test_binary)

y_train_pred = model.predict(x_train_binary)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"train accuracy is {train_accuracy}")

# Evaluate accuracy (proportion of right answers to false answers)
#accuracy = np.mean(y_pred == y_test)
#print(f"BNB Accuracy: {accuracy * 100:.2f}%")

# Evaluate accuracy using Scikit-learn
accuracy = accuracy_score(y_test, y_pred)
print(f"BNB Test Accuracy: {accuracy * 100:.2f}%")

# Generate and display the classification report
#print("\nClassification Report:")
#print(classification_report(y_test, y_pred))

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Find amount of TP, FP, FN, TN
num_classes = len(conf_matrix)
sum = 0
for i in range(num_classes):
    TP = conf_matrix[i, i]
    FP = conf_matrix[:, i].sum() - TP
    FN = conf_matrix[i, :].sum() - TP
    TN = conf_matrix.sum() - (TP + FP + FN)
    sum = TP + FP + FN + TN
    print(f"Class {i}: TP={TP}, FP={FP}, FN={FN}, TN={TN}")
    
print(f"total number of testing samples {sum}")

# Visualize the confusion matrix using Seaborn heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes, yticklabels=model.classes)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

report = classification_report(y_test, y_pred, output_dict=True)

# Function to plot classification report with support
def plot_classification_report_with_support(report):
    labels = list(report.keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
    metrics = ['precision', 'recall', 'f1-score', 'support']
    data = np.array([[report[label][metric] for metric in metrics] for label in labels])
    fig, ax = plt.subplots(figsize=(6, 8))
    cax = ax.matshow(data, cmap='coolwarm')
    plt.xticks(range(len(metrics)), metrics)
    plt.yticks(range(len(labels)), labels)
    plt.colorbar(cax)
    # Adding the text
    for (i, j), val in np.ndenumerate(data):
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white')
    plt.xlabel('Metrics')
    plt.ylabel('Classes')
    plt.title('Classification Report with Support')
    plt.show()

# Plotting the classification report with support
plot_classification_report_with_support(report)