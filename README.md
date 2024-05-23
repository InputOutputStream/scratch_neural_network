# 3-Layer Perceptron Model
Description
This model implements a 3-layer perceptron with an input layer, a hidden layer, and an output layer. It uses ReLU activation functions for the hidden layer and softmax for the output layer. The model is trained using stochastic gradient descent.

The main.py file contains the model class, and the note_book file allows for testing the functions separately.

# Features
Random initialization of weights and biases
ReLU and softmax activation functions
Calculation of model accuracy
Training the model with training data
Saving and loading the model from a file
Usage
Installation
Make sure you have Python installed on your system.

# Training the model

Create an instance of the model by specifying the sizes of the input, hidden, and output layers.
Call the fit method with the training data, the number of iterations, the learning rate, and other optional parameters.
The dataset, which is originally 42,000 lines, has been truncated for this occasion and is now 1,000 lines for testing and 200 for training.
Example:
python
Copy
from three_layer_perceptron import Model

# Create the model
model = Model(input_size=784, hidden_size=100, output_size=10)

# Load the training data
x_train = ...
y_train = ...

# Train the model
model.fit(x_train, y_train, eval=10, iters=100, a=0.1, show_training_info=True)
Using the trained model

Don't forget to update the file paths according to your needs

Use the predict method to get the model's predictions for new data.

Example:
# Predictions for new data
x_new = ...
predictions = model.predict(x_new)
Saving and loading the model
Use the save and load methods to save and load the model from a file.

# Example:

# Save the model
model.save('model.pkl')

# Load the saved model
loaded_model = Model.load('model.pkl')
A model.pkl file is provided in the directory to save you time on training, you can just load this model and make predictions.

Adapt the file paths and code examples based on your environment and data.
