import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow.keras as ks
import sklearn as sk
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input, decode_predictions
from keras.applications.xception import Xception
from PIL import ImageFile

# Authenticate
kagglehub.login()  # This will prompt you for your credentials.
# We also offer other ways to authenticate (credential file & env variables): https://github.com/Kaggle/kagglehub?tab=readme-ov-file#authenticate

# Download model
kagglehub.model_download('abdelghaniaaba/wildfire-prediction-dataset')

zip_path = 'wildfire-prediction-dataset.zip'

# Extract the contents
if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('content/datasets')  # Extract to a folder
    print("Files extracted to 'content/datasets'")
else:
    print(f"File {zip_path} not found!")

learning_rate = 0.001
size_inner = 100
drop_rate = 0.2

# Create model function


def create_model(learning_rate=0.001, size_inner=100, drop_rate=0.5):
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(350, 350, 3)
    )
    base_model.trainable = False

    inputs = ks.Input(
        shape=(350, 350, 3)
    )

    base = base_model(inputs, training=False)

    vectors = ks.layers.GlobalAveragePooling2D()(base)

    # adding inner layer
    inner = ks.layers.Dense(size_inner, activation='relu')(vectors)

    # dropout
    drop = ks.layers.Dropout(drop_rate)(inner)

    # 2 classes
    outputs = ks.layers.Dense(2)(drop)

    model = ks.Model(inputs, outputs)

    optimizer = ks.optimizers.Adam(learning_rate=learning_rate)

    loss = ks.losses.CategoricalCrossentropy(
        from_logits=True
    )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )

    return model


# Data augmentation
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    horizontal_flip=True,
    vertical_flip=True

    # I didn't use more props because the train process was 5 epochs = 1 hour... yes, amazing
    # shear_range=10,
    # rotation_range=30,
    # width_shift_range=30,
    # height_shift_range=10,
    # cval=0.0,
    # zoom_range=0.1,
)

train_ds = train_gen.flow_from_directory(
    '/content/datasets/train',
    target_size=(350, 350),
    batch_size=32
)

val_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

val_ds = val_gen.flow_from_directory(
    '/content/datasets/valid',
    target_size=(350, 350),
    batch_size=32,
    shuffle=False
)

# Create checkpoint
checkpoint = ks.callbacks.ModelCheckpoint(
    'xception_v1_{epoch:02d}_{val_accuracy:.3f}.keras',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'  # max monitor variable
)

# Ignore truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

model = create_model(
    learning_rate,
    size_inner,
    drop_rate
)

history = model.fit(
    train_ds, epochs=50,
    validation_data=val_ds,
    callbacks=[checkpoint]
)
