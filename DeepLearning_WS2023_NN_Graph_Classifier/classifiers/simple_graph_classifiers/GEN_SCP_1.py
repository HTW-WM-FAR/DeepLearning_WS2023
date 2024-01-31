
# %%
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import pandas as pd

from tensorflow import image
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.saving import save

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' ## Stops kernal error bug with matplotlib and tensorflow


# %%

DATASET_PATH = "C:\\Users\\pat_h\\htw_berlin_datasets\\GEN_SCP_1_DATASET"

BATCH_SIZE : int = 32
IMG_HEIGHT : int = 32
IMG_WIDTH : int = 32
VAL_SPLIT : int = 0.2
SEED : int = 123
EPOCHS : int = 3

# %%

train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=VAL_SPLIT,
    labels='inferred',
    subset='both',
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# %%

# Check to make sure data loaded correctly

class_names = train_ds.class_names
print(class_names)

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch)
    break

# %%

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# output layer size
num_classes = len(class_names)

model = Sequential([
    # Standardize values to be in the [0, 1] range by using tf.keras.layers.Rescaling
    layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])


# %%

##### compile model with choosen optimizer and loss function #####

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# %%

##### summary of model layers #####

model.summary()

# %%

##### fit model to train and val data #####

# returns History object
fitted_model = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# %%

# History.history is a dict with 'loss', 'accuracy', 'val_loss', 'val_accuracy'
print(fitted_model.history.keys())

# Note: metrics_names are available only after a keras.Model has been trained/evaluated on actual data.
print(model.metrics_names)

# %%

###### save model history to CSV #####

# choose file name and path
FILE_NAME = "GEN_SCP_1_HIST.csv"
FILE_PATH = "C:\\Users\\pat_h\\OneDrive\\Desktop\\public-repos\\NN-Graph-Classifier\\saved_models\\simple_graph_classifiers"

# create full path
save_data_path = os.path.join(FILE_PATH, FILE_NAME)

# %%

# save model history to CSV
model_history_df = pd.DataFrame(fitted_model.history)
print(model_history_df.head())

# %%

##### save history to CSV #####

model_history_df.to_csv(save_data_path)

# %%

##### save model to .keras file

# choose file name and path
FILE_NAME = "GEN_SCP_1.keras"
FILE_PATH = "C:\\Users\\pat_h\\OneDrive\\Desktop\\public-repos\\NN-Graph-Classifier\\saved_models\\simple_graph_classifiers"

# full save path
save_path = os.path.join(FILE_PATH, FILE_NAME)

model.save(save_path, save_format='keras')

# %%
