# %%

import matplotlib.pyplot as plt
import pandas as pd

import os


# %%

saved_evaluations = f"C:\\Users\\pat_h\\OneDrive\\Desktop\\public-repos\\NN-Graph-Classifier\\classifier_evaluations\\simple_classifiers"

eval_generated_false_count = pd.read_csv(os.path.join(saved_evaluations, f'eval_generated_false_count.csv'), header=0, names=['Dist', 'Count'])

eval_generated_false_count.head()
# %%


fig, ax = plt.subplots(1,1)
plt.bar(eval_generated_false_count['Dist'], eval_generated_false_count['Count'])
plt.title('Falsely Labeled Generated Graphs in CIFAR_SCP_1')
plt.xlabel('Distributions')
plt.ylabel('Counts')
fig.savefig('Falsely Labeled Generated Graphs in CIFAR_SCP_1')


# %%
