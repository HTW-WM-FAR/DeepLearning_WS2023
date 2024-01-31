# %%

import os
import pickle
import numpy as np 
from PIL import Image

# prepare the path for the first batch of pictures
cwd = os.getcwd()
batch_path = "C://Users//pat_h//htw_berlin_datasets//cifar-10_batches"
print(batch_path)


# %%

# download the first batch of pictures into a dictionary
file = os.path.join(batch_path, "data_batch_1")
with open(file, 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')

# %%
print(dict.keys())
print(f"Image Count Batch 1: {len(dict[b'data'])}")
print(f"Images: {dict[b'data'][0:10]}")
print(f"Images: {dict[b'filenames'][0:10]}")

# for image, file in dict[b'data'][0:10], dict[b'filenames'][0:10]:
#     print(image, file)

# %%

images_and_files = zip([np.array(image) for image in dict[b'data']], [file.decode('utf-8') for file in dict[b'filenames']])

print(type(images_and_files))

# for image, file in list(images_and_files)[0:3]:
#     print(image.shape, file)

for image, file in images_and_files:
    print(image, file)
    print(image.shape)
    break

# %%

# get first picture's data out of dictionary which is a 1-D arrary
first_image_test = np.array(dict[b'data'][0])
print(first_image_test.shape)
print(max(first_image_test), min(first_image_test))


# %%

# by spliting the rawdata into three arrays of (1024,0) I can
# then use column_stack to group the RGB together into (1024,3)
# from there it is easy to reshape it into the required (32,32,3) for RGB

for i in range(3):
    rawdata_r = np.array(dict[b'data'][i][:1024])
    rawdata_g = np.array(dict[b'data'][i][1024:2048])
    rawdata_b = np.array(dict[b'data'][i][2048:3072])

    reformed_data = np.column_stack((rawdata_r, rawdata_g, rawdata_b))
    reformed_data = np.reshape(reformed_data, (32,32,3))
    print(reformed_data.shape)

    pic = Image.fromarray(reformed_data, mode='RGB')
    pic.show()

# with this I can save the picture with its file name
# pic.save(dict[b'filenames'][0].decode("utf-8")) 

# %%

# create function to reshape image and save to new folder
cifar_save_path = os.path.join(os.getcwd(), 'graph dataset\\cifar')

if not os.path.exists(cifar_save_path):
    os.mkdir(cifar_save_path)

def cifar_reshape_save(image_data, file_names, save_path):

    zipped_data_file_pair = zip([np.array(image) for image in image_data], [file.decode('utf-8') for file in file_names])

    for image, file in zipped_data_file_pair:
        rawdata_r = image[:1024]
        rawdata_g = image[1024:2048]
        rawdata_b = image[2048:3072]

        reformed_data = np.column_stack((rawdata_r, rawdata_g, rawdata_b))
        reformed_data = np.reshape(reformed_data, (32,32,3))

        pic = Image.fromarray(reformed_data, mode='RGB')
        split_file = file.split('.')
        pic.save(os.path.join(save_path, split_file[0] + '.jpg'), format='JPEG')

cifar_reshape_save(image_data=dict[b'data'], file_names=dict[b'filenames'], save_path=cifar_save_path)


# %%

# why is this only temporary in the for loop itself?
# I assume the elements are expected to change only in the program
# since I didn't assign the value in respect to its position in the array
# for row in reformed_data[14]:
#     row = (0,255,0)

# it is now possible to make changes to the pixels directly
# this draws a horizontal blue line in the 16th row by setting the pixel RGB to (0,255,0)
for col in range(len(reformed_data[16])):
    reformed_data[16][col] = (0,255,0)

# by locking the column, we can draw a vertical red line (255,0,0) row by row 
print(f'Length of reformed_data: {len(reformed_data)}')
for row in range(len(reformed_data)):
    reformed_data[row][16] = (255,0,0)

pic = Image.fromarray(reformed_data, mode='RGB')
pic.show()

# %%

from PIL import Image
import numpy as np

# I can now investigate the charts I generate with matplotlib
im = Image.open("barchart_test.png")
a = np.asarray(im)

# the shape is (480, 640, 4)
# 480 height, 640 width, and RGB + alpha
print(f'Shape: {a.shape}\n', f'Size: {a.size}')
print(a[0].shape)
print(a[0][0].shape)
print(a[0][0][0].shape)

print(a[0])
print(a[0][0])
print(a[0][0][0])

