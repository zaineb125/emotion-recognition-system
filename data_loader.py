from keras.preprocessing.image import ImageDataGenerator


def data_loader(train_dir,val_dir,batch_size):
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=20,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       vertical_flip=True,
                                       fill_mode='nearest')

    val_datagen  = ImageDataGenerator(rescale = 1./255 )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=batch_size,
        class_mode='categorical')
    
    validation_generator = val_datagen.flow_from_directory(val_dir,
                                                  target_size = (48,48),
                                                  class_mode = 'categorical',
                                                  batch_size = 64)

    return train_generator,validation_generator