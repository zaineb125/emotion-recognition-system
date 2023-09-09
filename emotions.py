import argparse

import cv2
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.metrics import Precision
from keras.optimizers import Adam

import os
import mlflow
from sklearn.metrics import precision_score

from data_loader import data_loader
from model_history import plot_model_history
from frame_display import update_frame

import tkinter as tk

from confusion_matrix import plot_confusion_matrix

from create_model import model_creation
from simple_facerec import SimpleFacerec
from deepFace_model import resnet_model_creation

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--mode",help="train/display")
mode = ap.parse_args().mode


# Define data generators
train_dir = 'data/train'
val_dir = 'data/test'

num_train = 28709
num_val = 7178
batch_size = 124
num_epoch = 50


train_generator ,validation_generator = data_loader(train_dir,val_dir,batch_size)

# Define the learning rate scheduler
def learning_rate_schedule(epoch, lr):
    if epoch < 20:
        return lr
    elif epoch < 40:
        return lr * 0.1
    else:
        return lr * 0.01

# Create the model
model = model_creation()


#mlflow integration
mlflow.set_tracking_uri("mlruns")

# If you want to train the same model or try other models, go for this
if mode == "train":
    #artifact_path="Airline-Demo"
    with mlflow.start_run():
        model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])



        # Fit the model to the training data for one epoch
        model_info = model.fit_generator(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=50,
            validation_data=validation_generator,
            validation_steps=len(validation_generator)
        )

        model.save_weights('VGG16.h5')

        #print metrics
        y_true = validation_generator.classes
        y_pred = model.predict_generator(validation_generator, steps=num_val // batch_size).argmax(axis=-1)
        precision = precision_score(y_true, y_pred, average='weighted')
        print("Validation Precision:", precision)
        val_loss, val_accuracy = model.evaluate_generator(validation_generator, steps=num_val // batch_size)

        # Print the results
        print("Validation Loss:", val_loss)
        print("Validation Accuracy:", val_accuracy)


        # Log model metrics
        mlflow.log_metric("accuracy", model_info.history['accuracy'][-1])
        mlflow.log_metric("val_accuracy", model_info.history['val_accuracy'][-1])

        # Save the model
        mlflow.tensorflow.log_model(model, "models")
        plot_model_history(model_info)



# emotions will be displayed on your face from the webcam feed
elif mode == "display":

    # Load the pre-trained model weights
    model.load_weights('VGG16.h5')
    plot_confusion_matrix(model,validation_generator)

    # Dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # Create the GUI window
    window = tk.Tk()
    window.title("Emotion Detection")
    window.geometry("800x600")

    # Create a label to display the video feed
    video_label = tk.Label(window)
    video_label.pack()

    # Open the webcam
    cap = cv2.VideoCapture(0)

    # Start updating the video feed
    update_frame(cap, model, emotion_dict, video_label, window)


    # Start the GUI event loop
    window.mainloop()

    # Release the webcam
    cap.release()