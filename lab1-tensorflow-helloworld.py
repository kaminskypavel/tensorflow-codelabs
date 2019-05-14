# https://codelabs.developers.google.com/codelabs/tensorflow-lab1-helloworld/

import numpy as np
import tensorflow as tf
from tensorflow import keras

from serialize_utils import save_model_json, load_model_json

model = load_model_json('models/lab1-tensorflow-helloworld')
if (model == None):
    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')

    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

    model.fit(xs, ys, epochs=500)
    save_model_json(model, "models/lab1-tensorflow-helloworld")

y_predicted = model.predict([10.0])
print("the predicted value for 10.0 is %s" % y_predicted)
