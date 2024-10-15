#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import keras
import numpy as np
from keras import backend as K
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Input, Flatten
from keras.utils import plot_model
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
from matplotlib import pyplot as plt
from random import randint
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time


# In[5]:


# Define the MLP-3 architecture
model = Sequential()
model.add(Dense(784, activation='relu', input_shape=(784,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))


# In[6]:


# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[7]:


# Load and preprocess MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# make a copy before flattening to display a subset of the handwritten digits from the mnsist datatsets
X_train_copy = X_train


# In[8]:


# Flatten the images.
X_train = X_train.reshape((-1, 784))
X_test = X_test.reshape((-1, 784))

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalize the images
X_train /= 255
X_test /= 255

print(X_train.shape)           # (60000, 784)
print(X_test.shape)            # (10000, 784)


# In[9]:


# convert class vector to binary class matrices
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)


# In[7]:


for i in range(100):
    ax = plt.subplot(10, 10, i+1)
    ax.axis('off')
    plt.imshow(X_train_copy[randint(0, X_train.shape[0])], cmap='Greys')


# In[9]:


# Train the model
start_time = time.time()
training_log = model.fit(X_train, y_train, batch_size=50, epochs=50, validation_data=(X_test, y_test))
end_time = time.time()


# In[77]:


# Evaluating the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=False)

plt.plot(training_log.history['accuracy'])
plt.plot(training_log.history['val_accuracy'])
plt.title('model accuracy for mlp-3')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()

print(f'Test loss: {test_loss:.3}')
print(f'Test accuracy: {test_accuracy:.3}')


# In[45]:


# Calculate training time
training_time = end_time - start_time


# In[38]:


y_pred = model.predict(X_test)
y_pred_class = np.argmax(y_pred, axis=1)

y_true = np.argmax(y_test, axis=1)


# In[80]:


# Get the  MLP-3 test accuracy 
print(f'MLP-3 test accuracy: {test_accuracy:.3}''\n')

# Get the classification report 
print("Classification Report for mlp-3:")
print(classification_report(y_true, y_pred_class))


# In[76]:


# Get the confusion matrix
print("Confusion Matrix for mlp-3""\n")
print(confusion_matrix(y_true, y_pred_class))


# In[55]:


from sklearn.metrics import ConfusionMatrixDisplay

# Get the confusion matrix
cm = confusion_matrix(y_true, y_pred_class)

# Plot the confusion matrix
plt.figure(figsize=(12, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
disp.plot(include_values=True, cmap='Blues', ax=None)
plt.title('Confusion Matrix for mlp-3')
plt.show()


# In[81]:


# Print the metrics
print("Performance metrics for mlp-3:\n")
print(f'Test loss: {test_loss:.3}')
print(f'Test accuracy: {test_accuracy:.3}')
print(f'Training Time: {training_time:.6} seconds')


# In[73]:


# Select an image at random to feed to our model
idx = np.random.randint(len(X_test))

prediction = model.predict(X_test[[idx]])[0]

fig, ax = plt.subplots(2, 1)

ax[0].imshow(X_train_copy[idx], cmap='Greys')
ax[0].set_xticks(())
ax[0].set_yticks(())
ax[0].set_title('Image fed to model')

ax[1].bar(range(0, 10), prediction)
ax[1].set_xticks(range(0, 10))
ax[1].set_xlabel('Number')
ax[1].set_ylabel('Probability')
ax[1].set_title('Output probability distribution')

plt.show()
# dw2ve
print(f'Top prediction: {prediction.argmax()}')


# In[10]:


import joblib

# Saving Models
joblib.dump(model, 'model.pkl')

## load model
# my_model = joblib.load('model.pkl')

