import os
import sys

from tensorflow.python.estimator import keras


def save_model_json(model, model_dir):
    '''serializes a trained model to a folder on disk'''
    os.makedirs(model_dir, exist_ok=True)
    model.save("%s/model.h5" % model_dir)


def load_model_json(model_dir):
    '''loads a trained model from a folder on disk'''
    try:
        model = keras.models.load_model("%s/model.h5" % model_dir)
        return model
    except:
        print("Oops!", sys.exc_info(), "occured.")
        return None
