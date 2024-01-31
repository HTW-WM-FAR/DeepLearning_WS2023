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

##### show 3 x 3 image of graphs #####

# work_path = f'c:\\Users\\pat_h\\htw_berlin_datasets\\dist_datasets\\{GRAPH_DIST[INDEX]}'
# work_path = f'c:\\Users\\pat_h\\htw_berlin_datasets\\dist_datasets\\classifier_dataset_{GRAPH_DIM[INDEX_DIM]}\\{GRAPH_DIST[INDEX_DIST]}_{GRAPH_DIM[INDEX_DIM]}'

work_path = f'c:\\Users\\pat_h\\htw_berlin_datasets\\0Archive\\graph_dataset\\raw_data'

plt.figure(figsize=(10,8))
count = 1
for dirpath, dirnames, filenames in os.walk(work_path):
    while count < 10:
        for file in filenames[1000:1009]:
            plt.subplot(3, 3, count)
            im = Image.open(os.path.join(work_path, file))
            plt.imshow(im)
            plt.axis('off')
            # plt.title(str(file))
            count += 1


file_name = f"scraped_overview_original"
save_path = f"C:\\Users\\pat_h\\OneDrive\\Desktop\\public-repos\\NN-Graph-Classifier\\presentation\\images\\overviews\\{file_name}"

plt.suptitle(f'Scraped Graphs Original', fontsize=15.0, y=0.95)

plt.savefig(save_path)

# %%

##### generate overviews of all distribution graphs and their following dimensions #####

GRAPH_DIST = ['exp', 'lognorm', 'norm', 'unif']
GRAPH_DIM = ['32x32','115x86','153x115','460x345']

for graph_type in GRAPH_DIST:
    for graph_dim in GRAPH_DIM:
        
        work_path = f'c:\\Users\\pat_h\\htw_berlin_datasets\\dist_datasets\\classifier_dataset_{graph_dim}\\{graph_type}_{graph_dim}'

        plt.figure(figsize=(10,8))
        count = 1
        for dirpath, dirnames, filenames in os.walk(work_path):
            while count < 10:
                for file in filenames[0:9]:
                    plt.subplot(3, 3, count)
                    im = Image.open(os.path.join(work_path, file))
                    plt.imshow(im)
                    plt.axis('off')
                    plt.title(str(file))
                    count += 1


        file_name = f"{graph_type}_{graph_dim}_overview"
        save_path = f"C:\\Users\\pat_h\\OneDrive\\Desktop\\public-repos\\NN-Graph-Classifier\\presentation\\images\\overviews\\{file_name}"

        plt.suptitle(f'{graph_type.capitalize()} Distributions {graph_dim}', fontsize=15.0, y=0.95)

        plt.savefig(save_path)
        plt.close()
        
# %%


##### get overview for scraped graphs and natural images #####

FOLDER_NAMES = ['generated', 'natural', 'scraped']
INDEX = 2

work_path = f'C:\\Users\\pat_h\\htw_berlin_datasets\\CIFAR_GEN_SCP_1_DATASET\\{FOLDER_NAMES[INDEX]}'

file_name = f"{FOLDER_NAMES[INDEX]}_overview"
save_path = f"C:\\Users\\pat_h\\OneDrive\\Desktop\\public-repos\\NN-Graph-Classifier\\presentation\\images\\overviews\\{file_name}"

plt.figure(figsize=(10,8))
count = 1
for dirpath, dirnames, filenames in os.walk(work_path):
    while count < 10:
        for file in filenames[4000:4009]:
            plt.subplot(3, 3, count)
            im = Image.open(os.path.join(work_path, file))
            plt.imshow(im)
            plt.axis('off')
            # plt.title(str(file))
            count += 1

plt.suptitle(f'{FOLDER_NAMES[INDEX].capitalize()} 32x32', fontsize=15.0, y=0.95)

plt.savefig(save_path)
plt.close()

# %%
