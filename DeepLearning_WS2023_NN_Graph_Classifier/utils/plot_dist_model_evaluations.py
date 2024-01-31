# %%

import matplotlib.pyplot as plt
import pandas as pd

import os


# %%

##### plot the total amount of missclassifications per distribution graph #####

saved_evaluations = f"C:\\Users\\pat_h\\OneDrive\\Desktop\\public-repos\\NN-Graph-Classifier\\classifier_evaluations\\distribution_classifiers"

eval_false_count = pd.read_csv(os.path.join(saved_evaluations, f'eval_distribution_false_count.csv'), index_col=0)
eval_false_count.head()
list(eval_false_count.columns)
eval_false_count.loc[0]

# %%


fig, ax = plt.subplots(1,1)
plt.bar(list(eval_false_count.columns), eval_false_count.loc[0])
plt.title('Misclassified Distribution Graphs in DIST_153x115_1')
plt.xlabel('Distributions')
plt.ylabel('Counts')
fig.savefig('Misclassified Distribution Graphs in DIST_153x115_1')


# %%

##### plot the missclassifications per distribution graph #####

saved_evaluations = f"C:\\Users\\pat_h\\OneDrive\\Desktop\\public-repos\\NN-Graph-Classifier\\classifier_evaluations\\distribution_classifiers"

DIST_TYPE = ['exp', 'lognorm', 'norm', 'unif']

for dist in DIST_TYPE:

    eval_misclass_dist_df = pd.read_csv(os.path.join(saved_evaluations, f'{dist}_predictions_DIST_153x115_1.csv'), index_col=0)

    misclass_counts = eval_misclass_dist_df['label'].value_counts()

    fig, ax = plt.subplots(1,1)

    # use [1:] to ignore correct counts
    plt.bar(list(misclass_counts.keys()[1:]), misclass_counts[1:])

    plt.title(f'Misclassified Counts of {dist.capitalize()} in DIST_153x115_1')
    plt.xlabel('Distributions')
    plt.ylabel('Counts')

    fig.savefig(f'Misclassified Counts of {dist.capitalize()} in DIST_153x115_1')
    plt.close()

# %%

DIST_TYPE = ['exp', 'lognorm', 'norm', 'unif']

saved_evaluations = f"C:\\Users\\pat_h\\OneDrive\\Desktop\\public-repos\\NN-Graph-Classifier\\classifier_evaluations\\distribution_classifiers"

eval_misclass_dist_df = pd.read_csv(os.path.join(saved_evaluations, f'{DIST_TYPE[0]}_predictions_DIST_153x115_1.csv'), index_col=0)

eval_misclass_dist_df.loc[(eval_misclass_dist_df['prediction'] == False) & (eval_misclass_dist_df['label'] == 'lognorm') & (eval_misclass_dist_df['max'] > 9)]

# %%
