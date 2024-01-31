# https://matplotlib.org/stable/gallery/statistics/histogram_cumulative.html#sphx-glr-gallery-statistics-histogram-cumulative-py
# https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html

#%%

import numpy as np
import matplotlib.pyplot as plt
import os
import time

from utils.lorem_ipsum_prep import lorumipsum as lorumipsum

# %%

### generates and saves highly randomized graphs of the log-normal distribution ###

# measure script runtime
start_time = time.time()
# use lorem ipsum to avoid language bias
LORUMIPSUM = np.array(lorumipsum)
# init random generator
RNG = np.random.default_rng(12345)

# RGB color values for bins, lines, texts
rgb_bin = [] # bin color of histogram
rgb_line = [] # color of lines
rgb_face = [] # face color of figure
rgb_ax = [] # inside color of figure
rgb_xlabel = [] # color of text x label
rgb_ylabel = [] # color of text y label
rgb_title = [] # color of text title

RGB_WHITE = (1,1,1)
RGB_BLACK = (0,0,0)
GRAPHS = 20 # 2000 graphs: 305.011 seconds

# select save directory
SAVE_PATH = "C:\\Users\\pat_h\\htw_berlin_datasets\\dist_datasets\\lognorm"
# picture format
SAVE_DPI = 72
SAVE_FORMAT = 'jpg'

for graph in range(GRAPHS):
    
     # reset color RGB values for each iteration
    rgb_bin = []
    rgb_line = []
    rgb_face = []
    rgb_ax = []
    rgb_xlabel = []
    rgb_ylabel = []
    rgb_title = []

    fig, ax = plt.subplots()

    # randomize parameters
    mu = RNG.integers(-3.5,3.5) * RNG.random()  # mean of distribution
    sigma = RNG.integers(1,2.5) * RNG.random()  # standard deviation of distribution
    s = RNG.lognormal(mean=mu, sigma=sigma, size=RNG.integers(50,2000))
    num_bins = RNG.integers(15,100)

    # randomize color of bins
    for i in range(3):
        color = round(RNG.random(), 1)
        rgb_bin.append(color)

     # add alpha 0 to hide bins 50% of the time (only function line is visible)
    if RNG.random() > 0.5:
        rgb_bin.append(0)

    # the histogram of the data
    count, bins, ignored = plt.hist(s, num_bins, density=True, color=rgb_bin)

    # add a 'best fit' line
    x = np.linspace(min(bins), max(bins), 1000)
    y = np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi))

    # color of line
    for i in range(3):
        color = round(RNG.random(), 1)
        rgb_line.append(color)

    # solid or dashed line
    if RNG.random() > 0.5:
        line_style = '-'
    else:
        line_style = '--'

    ax.plot(x, y, line_style, color=rgb_line, linewidth=RNG.random()*3)

    # fig face color
    for i in range(3):
        color = round(RNG.random(), 1)
        rgb_face.append(color)
    if rgb_face == RGB_BLACK:
        rgb_face == RGB_WHITE
    if RNG.random() > 0.6: # create more white graphs
        rgb_face = RGB_WHITE

    fig.set_facecolor(rgb_face)

    # ax face color
    for i in range(3):
        color = round(RNG.random(), 1)
        rgb_ax.append(color)
    if RNG.random() > 0.6:
        rgb_ax = RGB_WHITE

    ax.set_facecolor(rgb_ax)
    
    # x label rgb
    for i in range(3):
        color = round(RNG.random(), 1)
        rgb_xlabel.append(color)
    # y label rgb
    for i in range(3):
        color = round(RNG.random(), 1)
        rgb_ylabel.append(color)
    # title rgb
    for i in range(3):
        color = round(RNG.random(), 1)
        rgb_title.append(color)

    # set lorum ipsum labels
    ax.set_ylabel(' '.join(np.random.choice(LORUMIPSUM, size = RNG.integers(2,5), replace=False)), color=rgb_ylabel, fontsize=max(10, RNG.integers(20,30)*RNG.random()))
    ax.set_xlabel(' '.join(np.random.choice(LORUMIPSUM, size = RNG.integers(2,5), replace=False)), color=rgb_xlabel, fontsize=max(10, RNG.integers(20,30)*RNG.random()))
    ax.set_title(' '.join(np.random.choice(LORUMIPSUM, size = RNG.integers(2,5), replace=False)), color=rgb_title, fontsize=max(10, RNG.integers(20,30)*RNG.random()))

    # add grid and adjust x limit 50% of the time
    if RNG.random() > 0.5:
        plt.grid(color='k', linestyle='-', linewidth=RNG.random())

    # adjust x limit 50% of the time
    if RNG.random() > 0.5:
        plt.xlim(0)

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()

    # save graph to folder path
    fig.savefig(os.path.join(SAVE_PATH, f'norm_{graph}.{SAVE_FORMAT}'), dpi=SAVE_DPI)
    plt.close(fig) # save memory usage

end_time = time.time()
print(f'Runtime: {round(end_time - start_time, 3)} seconds')

# %%

def lognorm_graph_generator(num_graphs, save_path, save_dpi, save_format, lorumipsum):
    ### generates and saves highly randomized graphs of the log-normal distribution ###

    # measure script runtime
    start_time = time.time()
    # use lorem ipsum to avoid language bias
    lorumipsum = np.array(lorumipsum)
    # init random generator
    RNG = np.random.default_rng(12345)

    # RGB color values for bins, lines, texts
    rgb_bin = [] # bin color of histogram
    rgb_line = [] # color of lines
    rgb_face = [] # face color of figure
    rgb_ax = [] # inside color of figure
    rgb_xlabel = [] # color of text x label
    rgb_ylabel = [] # color of text y label
    rgb_title = [] # color of text title

    RGB_WHITE = (1,1,1)
    RGB_BLACK = (0,0,0)

    for graph in range(num_graphs):
        
        # reset color RGB values for each iteration
        rgb_bin = []
        rgb_line = []
        rgb_face = []
        rgb_ax = []
        rgb_xlabel = []
        rgb_ylabel = []
        rgb_title = []

        fig, ax = plt.subplots()

        # randomize parameters
        mu = RNG.integers(-3.5,3.5) * RNG.random()  # mean of distribution
        sigma = RNG.integers(1,2.5) * RNG.random()  # standard deviation of distribution
        s = RNG.lognormal(mean=mu, sigma=sigma, size=RNG.integers(50,2000))
        num_bins = RNG.integers(15,100)

        # randomize color of bins
        for i in range(3):
            color = round(RNG.random(), 1)
            rgb_bin.append(color)

        # add alpha 0 to hide bins 50% of the time (only function line is visible)
        if RNG.random() > 0.5:
            rgb_bin.append(0)

        # the histogram of the data
        count, bins, ignored = plt.hist(s, num_bins, density=True, color=rgb_bin)

        # add a 'best fit' line
        x = np.linspace(min(bins), max(bins), 1000)
        y = np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi))

        # color of line
        for i in range(3):
            color = round(RNG.random(), 1)
            rgb_line.append(color)

        # solid or dashed line
        if RNG.random() > 0.5:
            line_style = '-'
        else:
            line_style = '--'

        ax.plot(x, y, line_style, color=rgb_line, linewidth=RNG.random()*3)

        # fig face color
        for i in range(3):
            color = round(RNG.random(), 1)
            rgb_face.append(color)
        if rgb_face == RGB_BLACK:
            rgb_face == RGB_WHITE
        if RNG.random() > 0.6: # create more white graphs
            rgb_face = RGB_WHITE

        fig.set_facecolor(rgb_face)

        # ax face color
        for i in range(3):
            color = round(RNG.random(), 1)
            rgb_ax.append(color)
        if RNG.random() > 0.6:
            rgb_ax = RGB_WHITE

        ax.set_facecolor(rgb_ax)
        
        # x label rgb
        for i in range(3):
            color = round(RNG.random(), 1)
            rgb_xlabel.append(color)
        # y label rgb
        for i in range(3):
            color = round(RNG.random(), 1)
            rgb_ylabel.append(color)
        # title rgb
        for i in range(3):
            color = round(RNG.random(), 1)
            rgb_title.append(color)

        # set lorum ipsum labels
        ax.set_ylabel(' '.join(np.random.choice(lorumipsum, size = RNG.integers(2,5), replace=False)), color=rgb_ylabel, fontsize=max(10, RNG.integers(20,30)*RNG.random()))
        ax.set_xlabel(' '.join(np.random.choice(lorumipsum, size = RNG.integers(2,5), replace=False)), color=rgb_xlabel, fontsize=max(10, RNG.integers(20,30)*RNG.random()))
        ax.set_title(' '.join(np.random.choice(lorumipsum, size = RNG.integers(2,5), replace=False)), color=rgb_title, fontsize=max(10, RNG.integers(20,30)*RNG.random()))

        # add grid and adjust x limit 50% of the time
        if RNG.random() > 0.5:
            plt.grid(color='k', linestyle='-', linewidth=RNG.random())

        # adjust x limit 50% of the time
        if RNG.random() > 0.5:
            plt.xlim(0)

        # Tweak spacing to prevent clipping of ylabel
        fig.tight_layout()

        # save graph to folder path
        fig.savefig(os.path.join(save_path, f'norm_{graph}.{save_format}'), dpi=save_dpi)
        plt.close(fig) # save memory usage

    end_time = time.time()
    return f'Runtime: {round(end_time - start_time, 3)} seconds'

# %%
