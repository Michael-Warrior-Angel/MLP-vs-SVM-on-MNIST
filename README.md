INTRODUCTION:
This project presents a comparative study of Multilayer Perceptron (MLP) and Support Vector Machine (SVM) for image classification using the MNIST dataset.
Both models are implemented using Python, with Keras for MLP and Scikit-learn for SVM. The models are evaluated based on performance metrics such as
accuracy, precision, recall, F1-score, training time and computational loss. The MNIST dataset, consisting of 70,000 images of handwritten digits 
provides a robust testing ground for assessing the strengths and limitations of these models. Results show that MLP, with its deep learning
architecture, achieves superior performance across most metrics, while SVM offers faster training time. This study highlights the trade-offs
between computational efficiency and predictive accuracy when choosing models for image classification.

ARCHITECTURE:
Multilayer Perceptron (MLP-3)
The MLP architecture, referred to as MLP-3 due to its three layers, is designed to handle the complexity of image data, particularly its ability to
learn non-linear relationships betwee features. The structure is as follows:

Input Layer: 784 neurons, corresponding to the 28x28 pixel values from the MNIST images.
Hidden Layer 1: 512 neurons with ReLU activation function.
Hidden Layer 2: 256 neurons with ReLU activation function.
Output Layer: 10 neurons with softmax activation for multi-class classification:

Support Vector Machine (SVM)
SVM operates differently, relying on the concept of hyperplanes to separate data points in a high-dimensional space. The kernel for SVM is 
RBF which is ideal for image classification tasks.

RBF Kernel: A non-linear kernel that uses a Gaussian function to project the data into a higher-dimensional space

Tools and Libraries:
Python is used as the main programming language for implementing the models. 
The following libraries are used:
1. Keras: To build and train the MLP.
2. Scikit-learn: For implementing SVM and calculating various performance metrics.
3. NumPy: For data manipulation and handling arrays.
4. Matplotlib: For visualizing results.

RESULTS AND DISCUSSION:
Both MLP and SVM are evaluated using metrics such as: accuracy, precision, recall, F1-score, computational loss, and efficiency (training time).

MLP Performance Metrics
1. Test loss (0.203): This represents the average loss of the model on the test dataset. A lower loss value indicates better performance, as it means
the model is making fewer errors in its predictions.
2. Test accuracy (0.982): This indicates that the model correctly predicted 98.2% of the instances in the test set. It’s a good overall performance, suggesting the model is well-suited
for the given task.
3. Training Time (8545 seconds): This represents the time taken to train the model. It’s important to consider the training time when evaluating the model’s efficiency, especially for
large datasets or complex models.
4. Overall Assessment: The MLP-3 model achieved a high test accuracy of 98.2% with a relatively low test loss, indicating good performance on the test dataset. The training time of 8545 seconds (approximately 2.37 hours) is reasonable, but it might be considered long for some applications. The training time could be reduced by optimizing the model architecture, using more efficient hardware, or by using a GPU (graphical processing unit), or exploring techniques like transfer learning.

MLP showed superior performance across almost all metrics. With an accuracy of 98.2%, it outperformed SVM, which achieved 97.9% accuracy with the RBF kernel. This result can be
attributed to MLP’s ability to model complex, non-linear relationships through its multiple hidden layers. However, training time is where SVM shines, taking only 1028 seconds 
compared to MLP’s 8545 second. Which is about 8 times longer than SVM’s training time, giving us an 8:1 ratio (MLP’s to SVM’s training time). This 
difference highlights a critical trade-off between predictive performance and computational efficiency. For applications requiring rapid deployment or where computational resources
are limited, SVM may be preferable.

SVM Performance
The SVM model exhibited strong performance, while its accuracy was lower than that of MLP, the training time advantage makes it suitable for
scenarios where quick predictions are essential. SVM’s performance indicates its effectiveness in simpler classification tasks, but its limitations
become evident with more complex datasets. The results emphasize the importance of model selection based on specific application
requirements. MLP’s deeper architecture provides robustness in accuracy but demands more computational power, whereas SVM offers
quick training and inference at the cost of some accuracy.

SVM Performance Metrics
1. Hinge Loss (0.0526): Hinge loss maximizes the margin between the positive and negative classes while minimizing the number of misclassified samples. A hinge loss between 0.0
and 0.1 indicates excellent classification performance, where the model correctly classifies most samples with a margin close to 1. A hinge loss of 0.0526 is a relatively low value, indicating that the model is doing well in its classification task.
2. Accuracy (0.979): An accuracy of 0.95 or higher is considered excellent for MNIST, indicating that the model is performing very well.
3. Training time (1028.44 seconds): This training time which is approximately 17 minutes is a reasonable time for training an SVM model on
the MNIST dataset. A training time of 5 to 30 minutes for an RBF kernel is the range for training on the MNIST dataset. Even though this
time is still very short compared to MLP-3 training time (2.37 hours), the training time obtained for our model can be significantly
reduced by using a GPU (graphical processing unit).



Here is a snapshot pictures depicting the performace metrics of the models.
![classificiation_report_for_mlp_3__verion_2](https://github.com/user-attachments/assets/4ad9a4e0-3b6d-41ea-b373-8940b249f74e)
![model_accuracy_for_mlp_3](https://github.com/user-attachments/assets/9b98b330-6466-4a76-865f-8c1e8e2feb69)
![heatmap_of_confusion_matrix_mlp_3](https://github.com/user-attachments/assets/b5f2395c-027c-47cf-b127-fe7c73e64673)
![one_of_the_predictions_from_mlp_3](https://github.com/user-attachments/assets/f44d3d7e-b63f-497e-9e6b-2d6847bff832)
![another_prediction_of_the_mlp_3_model](https://github.com/user-attachments/assets/f8d6d629-c478-417e-bc1a-5db31f75f7bd)

![classification_report_svm](https://github.com/user-attachments/assets/63ebde7a-c4a9-4317-b3ca-711dc2e4c088)
![confusion_matrix_for_svm](https://github.com/user-attachments/assets/6189a97e-bbdd-4b0e-8ee3-d5ba8a876e5e)
![prediction_2](https://github.com/user-attachments/assets/f07cc921-12eb-4de1-8abd-69e872c6cdb1)
![prediction_1](https://github.com/user-attachments/assets/2bfadaf5-09c5-4b55-bbd0-c6f132f07712)


CONCLUSION:
In conclusion, this study highlights the strengths and weaknesses of MLP and SVM for image classification tasks using the MNIST dataset. The
analysis conducted in this research demonstrates that MLP consistently outperformed SVM in accuracy and other classification
metrics, showcasing its capability for complex pattern recognition. However, SVM remains a viable option for tasks requiring rapid processing,
particularly when handling simpler datasets. 

With God's will, future poject may explore hybrid approaches that leverage the strengths of both MLP and SVM.
Additionally, the integration of Convolutional Neural Networks (CNNs) into the comparative analysis could provide deeper
insights into performance dynamics across various image classification challenges. Investigating CNN architectures may reveal their
potential for significantly improving accuracy and efficiency, especially in more complex datasets beyond MNIST.



