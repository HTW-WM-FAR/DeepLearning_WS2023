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

##### prepare correct paths and file names #####

# input correct folder and file name

# dimensions
GRAPH_DIM = '32x32'
# GRAPH_DIM = '115x86'
# GRAPH_DIM = '153x115'

MODEL = 2

# SIMPLE_MODEL_FOLDER = "simple_graph_classifiers"
DIST_MODEL_FOLDER = "distribution_graph_classifiers"
# SIMPLE_HIST_FILE = "CIFAR_GEN_SCP_1_HIST.csv"
dist_hist_file = f"DIST_{GRAPH_DIM}_{MODEL}_HIST.csv"
# GRAPH_TITLE = 'Natural Images, Generated Graphs, and Scraped Graphs'
GRAPH_TITLE = f'Distribution Graphs {GRAPH_DIM} Model {MODEL} (norm, lognorm, exp, unif)'

print(dist_hist_file)

# file_name = f"{SIMPLE_HIST_FILE.split('.')[0]}_RESULTS"
# hist_path = f"C:\\Users\\pat_h\\OneDrive\\Desktop\\public-repos\\NN-Graph-Classifier\\saved_models\\{SIMPLE_MODEL_FOLDER}\\{SIMPLE_HIST_FILE}"
# save_path = f"C:\\Users\\pat_h\\OneDrive\\Desktop\\public-repos\\NN-Graph-Classifier\\presentation\\images\\results\\{file_name}"

file_name = f"{dist_hist_file.split('.')[0]}_RESULTS"
hist_path = f"C:\\Users\\pat_h\\OneDrive\\Desktop\\public-repos\\NN-Graph-Classifier\\saved_models\\{DIST_MODEL_FOLDER}\\{dist_hist_file}"

save_path = f"C:\\Users\\pat_h\\OneDrive\\Desktop\\public-repos\\NN-Graph-Classifier\\presentation\\images\\results\\{file_name}"

model_history = pd.read_csv(hist_path, index_col=0)

print(model_history.head(20))

# %%

acc = model_history['accuracy']
val_acc = model_history['val_accuracy']

loss = model_history['loss']
val_loss = model_history['val_loss']

epochs_range = range(len(model_history['accuracy']))

# %%

cifar_color = '-g'
scrapped_color = '-b'
generated_color = '-r'

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')


plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

# title of all subplots
plt.suptitle(GRAPH_TITLE)

save_path = f"C:\\Users\\pat_h\\OneDrive\\Desktop\\public-repos\\NN-Graph-Classifier\\presentation\\images\\results\\{file_name}"

# savefig() needs to be before show()
plt.savefig(save_path)

plt.show()

# %%


# %%
