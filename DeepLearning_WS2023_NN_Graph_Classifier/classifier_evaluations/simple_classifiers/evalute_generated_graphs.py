# %%

import numpy as np
from PIL import Image
import tensorflow as tf
import os
import pandas as pd

from tensorflow import image
from tensorflow import keras

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' ## Stops kernal error bug with matplotlib and tensorflow


# %%

# images to evalute
EVAL_IMAGE_FOLDER = 'generated'
# model to predict
MODEL_FILE_NAME = "CIFAR_SCP_1.keras"

load_file_path = f"C:\\Users\\pat_h\\OneDrive\\Desktop\\public-repos\\NN-Graph-Classifier\\saved_models\\simple_graph_classifiers\\{MODEL_FILE_NAME}"

print(load_file_path)

# %%

eval_dataset_path = f"C:\\Users\\pat_h\\htw_berlin_datasets\\CIFAR_GEN_SCP_1_DATASET\\{EVAL_IMAGE_FOLDER}"

eval_image_dataset = []

for dirpath, dirnames, filenames in os.walk(eval_dataset_path):
    print(dirpath)
    for file in filenames:
        with Image.open(os.path.join(dirpath, file)) as im:
            im_array = keras.utils.img_to_array(im, data_format=None, dtype=None)
        eval_image_dataset.append((file, im_array))

# %%

print(eval_image_dataset[0])

# %%
    
# convert array into tensor and then into tensor dataset
eval_image_tensors = map(tf.convert_to_tensor, [array for file, array in eval_image_dataset])

# %%

# convert map to list
eval_image_tensors = list(eval_image_tensors)
print(len(eval_image_tensors))
print(tf.get_static_value(eval_image_tensors[0]))

# %%

# covert list of tensors to Dataset
eval_image_tensors = tf.data.Dataset.from_tensors(eval_image_tensors)

# %%

model_path = f"C:\\Users\\pat_h\\OneDrive\\Desktop\\public-repos\\NN-Graph-Classifier\\saved_models\\simple_graph_classifiers\\{MODEL_FILE_NAME}"
saved_model = tf.keras.models.load_model(model_path)
BATCH_SIZE = 20

if saved_model:
    print(f'Model loaded: {MODEL_FILE_NAME}')


# %%

# use loaded model to predict image class
pred_new = saved_model.predict(eval_image_tensors, batch_size=BATCH_SIZE, verbose="auto", steps=None, callbacks=None)

# %%

# display first 20 predictions to confirm if it is working
pred_new = pd.DataFrame(pred_new, columns=['graph', 'natural'])
pred_new.head(n=20)

# %%

pred_new = pred_new.assign(prediction=pred_new.graph > pred_new.natural )
pred_new.head(n=20)

# %%

accuracy_check = pred_new['prediction'].value_counts()
print(accuracy_check)

# %%

accuracy_check_df = pd.DataFrame(accuracy_check)
accuracy_check_df.head()
accuracy_check_df.to_csv(f'{EVAL_IMAGE_FOLDER}_in_{MODEL_FILE_NAME}_predictions.csv')

# %%

print(f'Accuracy: {100 * accuracy_check[True]/(accuracy_check[True] + accuracy_check[False])} %') 

# %%


# %%

# create new tuple list so I can sort and save the files easier
eval_image_pred = list(zip([file for file, array in eval_image_dataset], [array for file, array in eval_image_dataset], pred_new['prediction']))
print(type(eval_image_pred))

# %%

print(len(eval_image_pred[0]))

# %%

for file, image, prediction in eval_image_pred:
    print(file)
    print(image.shape)
    print(prediction)
    break


# %%

save_path = "C:\\Users\\pat_h\\OneDrive\\Desktop\\public-repos\\NN-Graph-Classifier\\missclass_images\\generated"

for file, image, prediction in eval_image_pred:
    if prediction == False:
        with keras.utils.array_to_img(image) as im_false:
            im_false.save(os.path.join(save_path, file))


# %%

##### count the instances of falsely labeled graphs #####
DATASET = 'generated'
count_false_file_path = f"C:\\Users\\pat_h\\OneDrive\\Desktop\\public-repos\\NN-Graph-Classifier\\missclass_images\\{DATASET}"

dist_types_dict = {'exp':0, 'lognorm':0, 'norm':0, 'unif':0}

for dirpath, dirnames, filenames in os.walk(count_false_file_path):
    
    print(dirpath)

    for file in filenames:

        split_file = file.split('_')

        for key, value in dist_types_dict.items():

            if split_file[0] == key:

                dist_types_dict[key] += 1


for key, value in dist_types_dict.items():
    print(f'{key}: {value}')

# %%

new_df = pd.DataFrame.from_dict(dist_types_dict, orient='index', columns=['Count'])
new_df.head()
new_df.to_csv(f'eval_{DATASET}_false_count.csv')

# %%
