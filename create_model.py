from keras import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.applications import MobileNet, MobileNetV2
from keras.applications import VGG16
from tensorflow.python.keras.layers import GlobalAveragePooling2D


def model_creation():
  """
  model = Sequential()

  model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,3)))
  model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(1024, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(7, activation='softmax'))
  """
  """
  input_shape = (48, 48, 3)
  # Create MobileNetV2 base model
  base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
  # Add custom layers for facial expression detection
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(1024, activation='relu')(x)
  predictions = Dense(7, activation='softmax')(x)

  # Create the transfer learning model
  model = Model(inputs=base_model.input, outputs=predictions)
  """
  input_shape = (48, 48, 3)
  # Create MobileNetV2 base model
  base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
  # Add custom layers for facial expression detection
  x= base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(1024, activation='relu')(x)
  predictions = Dense(7, activation='softmax')(x)

  # Create the transfer learning model
  model = Model(inputs=base_model.input, outputs=predictions)
  
  return model