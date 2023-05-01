import numpy as np
import wave 
from keras import models
from matplotlib import pyplot as plt
from recorder import record_audio, terminate,combineframes
from spec import preprocess
import tensorflow as tf 
import pandas as pd 
import torch
import os

model_1 = models.load_model("saved") 
model_2 = models.load_model("saved_take2") 

class_labels = [ 'fuck', 'not_bad', 'shit']  # a list of class labels in the same order as the model output

import time
def predict_mic():   
    
    spec = record_audio()

    spec= preprocess(spec) # input: waveform and sample rate spec
  
    prediction_1 = model_1.predict(spec)
    print(prediction_1)
    prediction_2 = model_2.predict(spec)
    print(prediction_2)

    max_prediction_1 = np.max(prediction_1, axis=1)
    predicted_class_1 = np.argmax(prediction_1, axis=1)
    predicted_word_1 = class_labels[predicted_class_1[0]]

    max_prediction_2 = np.max(prediction_2, axis=1)
    predicted_class_2 = np.argmax(prediction_2, axis=1)
    predicted_word_2 = class_labels[predicted_class_2[0]]

    print("Predicted Word 1 is:", predicted_word_1, predicted_class_1)

    print("Predicted Word 2 is:", predicted_word_2, predicted_class_2)


if __name__ == "__main__":
    predict_mic()  
    terminate()     