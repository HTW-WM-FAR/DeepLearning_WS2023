# %%

import matplotlib.pyplot as plt
import pandas as pd

import os


# %%

saved_evaluations = f"C:\\Users\\pat_h\\OneDrive\\Desktop\\public-repos\\NN-Graph-Classifier\\classifier_evaluations\\distribution_test_evaluation"

eval_generated_false_count = pd.read_csv(os.path.join(saved_evaluations, 'test_evaluate_DIST_153x115_1.csv'), index_col=0)

eval_generated_false_count.head()
# %%

false_count_plot = eval_generated_false_count.loc[eval_generated_false_count['prediction']==False]['label'].value_counts()

# %%

fig, ax = plt.subplots(1,1)
plt.bar(list(false_count_plot.keys()), false_count_plot)
plt.title('Count of Misclassified Graphs in DIST_153x115_1')
plt.xlabel('Distributions')
plt.ylabel('Counts')
fig.savefig('Count of Misclassified Graphs in DIST_153x115_1')


# %%


eval_generated_false_count.loc[(eval_generated_false_count['actual']=='norm') & (eval_generated_false_count['prediction']==True)]

# %%

eval_generated_false_count.loc[(eval_generated_false_count['actual']=='lognorm') & (eval_generated_false_count['prediction']==False)]

# %%
