import numpy as np

from tensorflow.keras import models
import tensorflow as tf

from recording_helper import record_audio, terminate
from tf_helper import preprocess_audiobuffer

commands = ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']

loaded_model = tf.saved_model.load("saved")


def predict_mic():
    audio = record_audio()
    audio = preprocess_audiobuffer(audio)
    prediction = loaded_model(audio)
    # print(prediction)
    # label_pred = np.argmax(prediction['predictions'], axis=1)
    # command = commands[label_pred[0]]
    command = prediction['class_names']
    print("Predicted label:", command)
    return command


if __name__ == "__main__":
    from turtle_helper import move_turtle
    while True:
        command = predict_mic()
        move_turtle(command)
        if command == "stop":
            terminate()
            break
