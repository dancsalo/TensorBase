## NNClasses: Neural Network Classes for Tensorflow

Python 3 classes created to streamline machine learning applications in Tensorflow. Similar in concept to Keras and PrettyTensor but more compact and hopefully more underestandable. More example networks to be added in the future.

### Classes:
* **Model**: a parent class for Tensorflow models (NNModel.py).
* **Layers**: a parent class for connected and convolutional networks (NNLayers.py).
* **Data**: a parent class for batch-generating during training and testing (NNData.py).

### Documentation:
* The Template folder contains commented Data and Models objects.
* The Example folder contains a convolutional Variational Autoencoder (Model.py) that runs on the MNIST dataset (Data.py). Tested 12/1/16 on Ubuntu 14.04, CUDA 8.0, cuDNN 5.1, Tensorflow 0.12.0.