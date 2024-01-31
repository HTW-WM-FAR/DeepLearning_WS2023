# %%

import numpy as np
from PIL import Image
import tensorflow as tf
import os
import pandas as pd

from tensorflow import image
from tensorflow import keras

import random
random.seed(123)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' ## Stops kernal error bug with matplotlib and tensorflow


# %%

##### Load model to predict image class #####

# model to predict
MODEL_FILE_NAME = "DIST_153x115_1.keras"

load_model_file_path = f"C:\\Users\\pat_h\\OneDrive\\Desktop\\public-repos\\NN-Graph-Classifier\\saved_models\\distribution_graph_classifiers\\{MODEL_FILE_NAME}"

print(load_model_file_path)

saved_model = tf.keras.models.load_model(load_model_file_path)
BATCH_SIZE = 20 # for prediction later

if saved_model:
    print(f'Model loaded: {MODEL_FILE_NAME}')


# %%

##### fill each key with list of file name image tensor array tuples #####
    
eval_dataset_path = f"C:\\Users\\pat_h\\htw_berlin_datasets\\test_eval_datasets\\all_153x115"
file_image_arr = []

for dirpath, dirnames, filenames in os.walk(eval_dataset_path):

    print(dirpath)

    for file in filenames:
                
        with Image.open(os.path.join(dirpath, file), formats=['JPEG']) as im:

            im_array = keras.utils.img_to_array(im, data_format=None, dtype=None)

            # this dictionary will remain with simple array for convert to image later
            # im_array = tf.convert_to_tensor(im_array)

        file_image_arr.append((file, im_array))

# %%
            
print(file_image_arr[0])
print(file_image_arr[0][1].shape)

# %%

##### convert array into tensor and then into tensor dataset #####

image_tensors = []

for file, array in file_image_arr:

    tensor_array = tf.convert_to_tensor(array)

    image_tensors.append(tensor_array)
    
    # convert list of tensor data into TensorDataset

image_tensors_datatset = tf.data.Dataset.from_tensors(image_tensors)
type(image_tensors_datatset)

# %%

##### use loaded model to predict image class #####

pred_results = saved_model.predict(image_tensors_datatset, batch_size=BATCH_SIZE, verbose="auto", steps=None, callbacks=None)

pred_results = pd.DataFrame(pred_results, columns=['exp', 'lognorm', 'norm', 'unif']) # direct from model's train_ds.class_names

pred_results.head()

# %%

##### create new max columun in pred_results #####

# add max column between each prediction
pred_results = pred_results.assign(max=pred_results[list(pred_results.keys())].max(axis=1))
pred_results.head()



# %%

##### merge the file name into the prediction dictionary #####

    
file_name_list = [file_name for file_name, image_array in file_image_arr]

file_name_df = pd.DataFrame(file_name_list, columns=['file_name'])
   
pred_results = pred_results.merge(file_name_df, how='inner', left_index=True, right_index=True)

pred_results.head()

# %%

                
##### add the predicted label to label column #####

dist_labels = ['exp', 'lognorm', 'norm', 'unif'] 

# create new column for label
pred_results = pred_results.assign(actual='actual')
pred_results = pred_results.assign(label='label')
pred_results = pred_results.assign(prediction='bool')

# run once for entire df
for i in range(len(pred_results)):

    # get slice of df at row i
    slice = pred_results.iloc[i]

    # for every slice of df (row) go through each distribution type
    # if the max of the row is equal to the num in exp, lognorm, norm, or unif row then set label to corresponding distribution
    for dist in dist_labels:    
        if slice.loc['max'] == slice.loc[dist]:

        # use df.loc[] for insertions into df
            pred_results.loc[i, 'label'] = dist
    
    file = pred_results.loc[i, 'file_name']
    file_dist = file.split('_')[0]
    pred_results.loc[i, 'actual'] = file_dist
    # print(file_dist)
    # print(file_dist ==  pred_results.loc[i, 'label'])
    pred_results.loc[i, 'prediction'] = file_dist == pred_results.loc[i, 'label']

pred_results.head(n=10)

# %%

pred_results.to_csv(f'test_evaluate_DIST_153x115_1.csv')

# %%

pred_results.loc[pred_results['prediction']==False]['label'].value_counts()

# %%

pred_results.loc[(pred_results['prediction']==False) & (pred_results['actual']=='unif')]
# %%

pred_results.loc[(pred_results['actual']=='norm') & (pred_results['prediction']==False)]

# %%

