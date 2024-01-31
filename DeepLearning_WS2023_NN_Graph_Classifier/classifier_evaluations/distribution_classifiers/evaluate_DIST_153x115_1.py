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
    
dist_image_dict = {'exp':[], 'lognorm':[], 'norm':[], 'unif':[]}

eval_dataset_path = f"C:\\Users\\pat_h\\htw_berlin_datasets\\DIST_153x115_1_DATASET"

for dirpath, dirnames, filenames in os.walk(eval_dataset_path):

    print(dirpath)

    for file in filenames:

        for dist_type in dist_image_dict.keys():

            if os.path.basename(dirpath) == dist_type:
                
                with Image.open(os.path.join(dirpath, file)) as im:

                    im_array = keras.utils.img_to_array(im, data_format=None, dtype=None)

                    # this dictionary will remain with simple array for convert to image later
                    # im_array = tf.convert_to_tensor(im_array)

                dist_image_dict[dist_type].append((file, im_array))


# %%

##### convert array into tensor and then into tensor dataset #####

dist_image_dict_tensors = {}

for key, value in dist_image_dict.items():

    # create separate dictionary only for tensor data
    dist_image_dict_tensors[key] = []

    for file, array in value:

        tensor_array = tf.convert_to_tensor(array)

        dist_image_dict_tensors[key].append(tensor_array)
    
    # convert list of tensor data into TensorDataset
    dist_image_dict_tensors[key] = tf.data.Dataset.from_tensors(dist_image_dict_tensors[key])


# %%

##### use loaded model to predict image class #####

pred_dict = {}

for key, value in dist_image_dict_tensors.items():
   
    pred_dict[key] = saved_model.predict(value, batch_size=BATCH_SIZE, verbose="auto", steps=None, callbacks=None)

    pred_dict[key] = pd.DataFrame(pred_dict[key], columns=['exp', 'lognorm', 'norm', 'unif']) # direct from model's train_ds.class_names


# %%

##### create dictionary with accuracy and create new max columun in pred_dict #####

accuracy_check_dict = {}

for key, df in pred_dict.items():

    # add prediction True or False in each key (distribution type)
    pred_dict[key] = df.assign(prediction=df[list(pred_dict.keys())].max(axis=1) == df[key])

    # add max column between each prediction
    pred_dict[key] = df.assign(max=df[list(pred_dict.keys())].max(axis=1))

    accuracy_check_dict[key] = pred_dict[key].prediction.value_counts()


# %%

##### display and then save number of misclassfications per distribution #####

false_pred = 0 
false_count = []

for key, value in accuracy_check_dict.items():
    print(f'--------{key}--------')
    false_pred += value.loc[False]
    false_count.append(value.loc[False])
    value.loc['Accuracy'] = value.loc[True] / (value.loc[True] + value.loc[False]) 
    print(value.head(n=5))
    
false_count_df = pd.DataFrame(np.atleast_2d(false_count), columns=list(accuracy_check_dict.keys()))
false_count_df.head()
false_count_df.to_csv('eval_distribution_false_count.csv')


# %%

##### merge the file name into the prediction dictionary #####

for key_1, tup in dist_image_dict.items():
    
    file_name_list = [file_name for file_name, image_array in tup]

    file_name_df = pd.DataFrame(file_name_list, columns=['file_name'])
   
    for key_2, df in pred_dict.items():

        if key_1 == key_2:

            pred_dict[key_2] = df.merge(file_name_df, how='inner', left_index=True, right_index=True)

for key, df in pred_dict.items():

    print(pred_dict[key].head(n=5))

             
# %%

##### create new dictionary with file names, predictions, and image arrays #####
    
evaluate_file_pred_array_dict = {}

for key_1, tup in dist_image_dict.items():
    
    file_name_list = [file_name for file_name, image_array in tup]
    image_array_list = [image_array for file_name, image_array in tup]

    for key_2, df in pred_dict.items():

        pred_list = df.prediction

        if key_1 == key_2:

            evaluate_file_pred_array_dict[key_1] = list(zip(file_name_list, pred_list, image_array_list))


# %%

##### save the misclassified images to their respective folder #####
            
save_path = "C:\\Users\\pat_h\\OneDrive\\Desktop\\public-repos\\NN-Graph-Classifier\\missclass_images\\DIST_153x115"

for key, tup in evaluate_file_pred_array_dict.items():

    for file, pred, image in tup:
        
        dist_type = file.split('_')[0]
        
        if pred == False:
            with keras.utils.array_to_img(image) as im_false:
                im_false.save(os.path.join(save_path, dist_type, file))


# %%
                
##### add the predicted label to label column #####
                
for key_1, df in pred_dict.items():
    
    # create new column for label
    pred_dict[key_1] = df.assign(label='dist')

    # run once for entire df in key
    for i in range(len(df)):

        # get slice of df at row i
        slice = df.iloc[i]

        # for every slice of df (row) go through each distribution type
        for key_2 in pred_dict.keys():

            # if the max of the row is equal to the num in exp, lognorm, norm, or unif row then set label to corresponding distribution
            if slice.loc['max'] == slice.loc[key_2]:
                
                # use df.loc[] for insertions into df
                pred_dict[key_1].loc[i, 'label'] = key_2
        


# %%

for key, df in pred_dict.items():
    
    df.to_csv(f'{key}_predictions_DIST_153x115_1.csv')
# %%
