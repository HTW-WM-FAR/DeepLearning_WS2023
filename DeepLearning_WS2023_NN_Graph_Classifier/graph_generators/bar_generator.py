
# %%

import numpy as np
import matplotlib.pyplot as plt
import random
import os

from utils.lorem_ipsum_prep import lorumipsum as lorumipsum

# %%

### generates and saves highly randomized bar graphs ###

# use lorem ipsum to avoid language bias
words = np.array(lorumipsum)
rng = np.random.default_rng()

rand_rgb = [] # randomized rgb
rgb_list = [] # list of rgb values
rgb_face = [] # face color of figure
rgb_ax = [] # inside color of figure

rgb_white = (1,1,1)
rgb_black = (0,0,0)
graph_amount = 50

for graph in range(graph_amount):
    
    # reset values for each iteration
    rand_rgb = []
    rgb_list = []
    rgb_face = []
    rgb_ax = []

    fig, ax = plt.subplots()

    # determine number of bars
    element_count = rng.integers(2,8)

    # rgb values for each bar in graph
    for col in range(element_count):

        rand_rgb = []

        for i in range(3):

            color = round(random.random(), 1)
            rand_rgb.append(color)

        rand_rgb = tuple(rand_rgb)
        rgb_list.append(rand_rgb)

    words = rng.choice(lorumipsum, size = element_count, replace=False)
    counts = rng.integers(low=0, high=500, size=element_count)

    bar_labels = words
    bar_colors = rgb_list

    # fig face color
    for i in range(3):

        color = round(random.random(), 1)
        rgb_face.append(color)

    rgb_face = tuple(rgb_face)

    rand_int = random.random()

    if (rgb_face in rgb_list) or (rgb_face == rgb_black):
        rgb_face = rgb_white
    if rand_int > 0.6: # create more white graphs
        rgb_face = rgb_white

    fig.set_facecolor(rgb_face)

    ax.bar(words, counts, label=bar_labels, color=bar_colors)

    # ax face color
    for i in range(3):

        color = round(random.random(), 1)
        rgb_ax.append(color)

    rgb_ax = tuple(rgb_ax)
    
    rand_int = random.random()

    if rgb_ax in rgb_list:
        rgb_ax = rgb_white
    if rand_int > 0.6:
        rgb_ax = rgb_white

    ax.set_facecolor(rgb_ax)

    # set lorum ipsum labels
    ax.set_ylabel(' '.join(np.random.choice(lorumipsum, size = rng.integers(2,5), replace=False)))
    ax.set_xlabel(' '.join(np.random.choice(lorumipsum, size = rng.integers(2,5), replace=False)))
    ax.set_title(' '.join(np.random.choice(lorumipsum, size = rng.integers(2,5), replace=False)))

    fig.savefig(os.path.join('test_graphs', 'bar', f'bar_{graph}.jpg'))
    plt.close(fig) # save memory usage


# %%
