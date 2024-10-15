#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from keras.datasets import mnist
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import time
import matplotlib.pyplot as plt
from random import randint


# In[2]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Flatten the images into 1D vectors 
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))

# Normalize the pixel values (important for SVM as well)
x_train = x_train / 255.0 - 0.5
x_test = x_test / 255.0 - 0.5


# In[8]:


class_labels, class_freq = np.unique(y_train, return_counts=True)

# Print the class labels and their frequencies 
print('Class Labels:', class_labels)
print('Class Frequencies:', class_freq)

# Plot a bar chart to visualize the class frequencies 
plt.bar(class_labels, class_freq)
plt.xlabel('Class_label')
plt.ylabel('Frequency')
plt.title("Class Frequencies in MNIST Dataset")

# # Show numbers above bars
# for i, freq in enumerate(class_freq):
#     plt.text(i, freq + 100, str(freq), ha='center', va='bottom')
    
plt.show()


# In[4]:


# Initialize the SVM model with RBF kernel (good for classification tasks like MNIST)

svm_classifier = SVC(kernel='rbf', gamma='scale')  # gamma='scale' is a good default for image classification

# Train the SVM model
start_time = time.time()
svm_classifier.fit(x_train, y_train)
end_time = time.time()

training_time = end_time - start_time


# In[10]:


# Predict on the test set
y_pred = svm_classifier.predict(x_test)


# In[11]:


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy and classification report
print(f'SVM Test Accuracy: {accuracy:.3}''\n')
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[6]:


# Step 5: Plot Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix for SVM')
plt.colorbar()
tick_marks = np.arange(10)
plt.xticks(tick_marks, range(10))
plt.yticks(tick_marks, range(10))

# Plotting text inside matrix boxes
thresh = conf_matrix.max() / 2.
for i, j in np.ndindex(conf_matrix.shape):
    plt.text(j, i, format(conf_matrix[i, j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[7]:


# Assuming the number of classes
num_classes = conf_matrix.shape[0]

# Initialize variables to store the results
true_positives = np.zeros(num_classes)
false_positives = np.zeros(num_classes)
false_negatives = np.zeros(num_classes)
true_negatives = np.zeros(num_classes)

# Iterate over the classes
for i in range(num_classes):
    # calculate true positives, false positives, and false negatives for each class
    true_positives[i] = conf_matrix[i, i]
    false_positives[i] = sum(conf_matrix[:, i]) - true_positives[i]
    false_negatives[i] = sum(conf_matrix[i, :]) - true_positives[i]
    
# calculate true negatives for each class
true_negatives[i] = sum(sum(conf_matrix)) - sum(conf_matrix[:, i]) - sum(conf_matrix[i, :]) + conf_matrix[i,i]               
# true_negatives[i] = sum(sum(conf_matrix)) - false_positives[i] - false_negatives[i] + 3*true_positives[i]


# Print the results for each class
for i in range(num_classes):
    print(f"class {i}:")
    print(f"Count of number {i}: {class_counts[i]}")  
    print(f"True positives: {true_positives[i]}")
    print(f"False negatives: {false_negatives[i]}")
    print(f"False positives: {false_positives[i]}")
    # print(f"True negatives: {true_negatives[i]}")
    print("=========================\n")
    
print(f"Total Predictions: {sum(sum(conf_matrix))}")

# # calculate the count of members within each class
# class_counts = np.sum(conf_matrix, axis=1)
# print("class counts:")
# for i in range(num_classes):
#     print(f"class {i}: {class_counts[i]}")


# In[8]:


# Select a random image from the test set
idx = np.random.randint(len(x_test))
image = x_test[idx]
label = y_test[idx]

# Predict the label
y_pred_label = svm_classifier.predict([image])[0]

# Display the image and the prediction
plt.imshow((image + 0.5).reshape(28, 28), cmap='Greys')
plt.title(f"True label: {label}, Predicted: {y_pred_label}")
plt.axis('off')
plt.show()


# In[34]:


from sklearn.metrics import hinge_loss
# Calculate hinge loss
hinge_loss_value = hinge_loss(y_test, svm_classifier.decision_function(x_test))


# In[41]:


print("Performance metrics for SVM:\n" )

print(f'Hinge Loss: {hinge_loss_value:.3}')
print(f'Accuracy: {accuracy:.3}')
print(f'Traning Time: {training_time:.6} seconds')


# In[45]:


import joblib

# Saving Model
joblib.dump(svm_classifier, 'svm_classifier.pkl')

# load model
# my_model = joblib.load('svm_classifier.pkl')

