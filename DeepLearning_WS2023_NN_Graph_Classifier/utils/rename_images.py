# %%

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time



# %%

##### check folder and file names #####

work_path = 'c:\\Users\\pat_h\\htw_berlin_datasets\\dist_datasets'

for dirpath, dirnames, filenames in os.walk(work_path):
    print('Top Folder: ' + os.path.basename(dirpath))
    for name in dirnames:
        print('\t' + 'Sub Folder: ' + name)
    for file in filenames[0:4]:
        print('\t' + 'File: ' + file + '\t' + str(os.path.getsize(os.path.join(dirpath, file))/1000) + ' KB')

# %%

##### ACCIDENTALLY SAVED THE WRONG FILE NAMES SO I NEEDED TO RENAME THEM ######
# rename all files to their correct distribution
start_time = time.time()

work_path = 'c:\\Users\\pat_h\\htw_berlin_datasets\\dist_datasets'

# walk through dir
for dirpath, dirnames, filenames in os.walk(work_path):
    print(f'PATH BASE: {os.path.basename(dirpath)}')

    # resize images and save to new dir 'resized'
    for file in filenames:
        file_number = file.split('_')[1]
        new_name = f'{os.path.basename(dirpath)}_{file_number}'
        os.rename(os.path.join(dirpath, file), os.path.join(dirpath, new_name))

end_time = time.time()
print(f'Total Runtime: {round(end_time - start_time, 3)} seconds') # Total Runtime: 1.496 seconds

# %%
