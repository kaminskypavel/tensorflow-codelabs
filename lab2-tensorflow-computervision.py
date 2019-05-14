import tensorflow as tf

print(tf.__version__)

mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
import matplotlib.pyplot as plt

plt.imshow(training_images[0])

training_images = training_images / 255.0
test_images = test_images / 255.0

# Sequential: This defines a SEQUENCE of layers in the neural network.
#
# Flatten: Remember earlier, our images were a square when printed them out.
# Flatten just takes that square and turns it into a one-dimensional vector.
#
# Dense: Adds a layer of neurons.
#
# Each layer of neurons needs an activation function to tell them what to do.
# There are lots of options, but use these for now.
#
# Relu effectively means if X>0 return X, else return 0.
# It only passes values 0 or greater to the next layer in the network.
#
# Softmax takes a set of values, and effectively picks the biggest one.
# For example, if the output of the last layer looks like
# [0.1, 0.1, 0.05, 0.1, 9.5, 0.1, 0.05, 0.05, 0.05],
# it saves you from saving to sort for the largest value—it returns
# [0,0,0,0,1,0,0,0,0].


model = tf.keras.models.Sequential(
    [tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(128, activation=tf.nn.relu),
     tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer = tf.train.AdamOptimizer(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)

print(model.evaluate(test_images, test_labels))
