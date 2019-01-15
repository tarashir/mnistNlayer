# Running the code:
1) run ConvolutionalPreprocessing.py.
2) You should now be able to run MNIST_nLayer and MNIST_CNN :)

NOTE: I apologize for the bad coding style.  I coded these back when I was still "new" to coding and did not write them with the intention of looking back on them or having others read them :(.  Also, most 

Other than a few lines of code (specified in each file), all code in nLayer and CNN were written by mine.  No API's were used.
If you run the code, you probably won't achieve the accuracies mentioned below because 1) I don't know what batch size and number of iterations I used 2) I have these values set low so that the files don't take too long to run.

# Descriptions of the files:

mnist: library I use only for reading the mnist files.  This is not my code but unfortunately I can't find the library I used anymore (I think it was updated b/c I can't find the one I used).

ConvolutionalPreprocessing: Used to preprocess images and transform layers for the CNN.  Uses techniques such as center of mass shifts, convolutions, template matching, RELU, pooling/max pooling, and a form of edge detection a 7 year old could probably write.

MNIST_CNN: A Convolutional Neural Net that I achieved 93.77% accuracy on the test data with.  Trained using Genetic Algorithm.

MNIST_nLayer: A straight forward Vanilla Neural Net trained using batch gradient descent.  Achieved an accuracy of 85.5%. Just by changing values, the user can alter both the number of layers and the dimensions of each layer.  Uses RELU, dropout, Xavier initialization, learning rate decay, max-norm Regularization 
