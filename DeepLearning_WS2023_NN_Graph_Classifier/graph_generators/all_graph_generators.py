# %%

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import multiprocessing as mp

from utils.lorem_ipsum_prep import lorumipsum as LORUMIPSUM

# %%

def unif_graph_generator(save_path:str, num_graphs=100, save_dpi=72, save_format='jpg', lorumipsum=LORUMIPSUM):
    ### generates and saves highly randomized graphs of the uniform distribution ###

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

        num_bins = RNG.integers(20,80)

        # randomize color of bins
        for i in range(3):
            color = round(RNG.random(), 1)
            rgb_bin.append(color)

        # add alpha 0 to hide bins 50% of the time
        if RNG.random() > 0.5:
            rgb_bin.append(0)

        # the histogram of the data
        a = RNG.integers(-5,-1)
        b = RNG.integers(0,5)
        
        s = RNG.uniform(a,b,1000)
        count, bins, ignored = plt.hist(s, num_bins, density=True, color=rgb_bin)

        # add a 'best fit' line
        y = 1 /(b - a)

        # color of line
        for i in range(3):
            color = round(RNG.random(), 1)
            rgb_line.append(color)

        # solid or dashed line
        if RNG.random() > 0.5:
            line_style = '-'
        else:
            line_style = '--'

        ax.plot(bins, np.ones_like(bins) * y, line_style, color=rgb_line, linewidth=RNG.random()*3)

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

        # add grid 50% of the time
        if RNG.random() > 0.5:
            plt.grid(color='k', linestyle='-', linewidth=RNG.random())

        plt.vlines(a, ymin=0, ymax=y, colors=rgb_line, linestyles=line_style)
        plt.vlines(b, ymin=0, ymax=y, colors=rgb_line, linestyles=line_style)

        # Tweak spacing to prevent clipping of ylabel
        fig.tight_layout()

        fig.savefig(os.path.join(save_path, f'unif_{graph}.{save_format}'), dpi=save_dpi)
        plt.close(fig) # save memory usage

    # calculate runtime
    end_time = time.time()
    return f'Runtime: {round(end_time - start_time, 3)} seconds'

# %%

def exp_graph_generator(save_path:str, num_graphs=100, save_dpi=72, save_format='jpg', lorumipsum=LORUMIPSUM):
    ### generates and saves highly randomized graphs of the exponetial distribution ###

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

        # example data
        lamb = RNG.random() + RNG.random()  # lambda
        s = RNG.exponential(scale=lamb, size=RNG.integers(50,2000))
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
        x = np.linspace(min(bins), max(bins))
        y = lamb*np.exp(-lamb*x)

        # color of line
        for i in range(3):
            color = round(RNG.random(), 1)
            rgb_line.append(color)

        # solid or dashed line
        if RNG.random() > 0.5:
            line_style = '-'
        else:
            line_style = '--'

        # plot line
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

        # add grid 50% of the time
        if RNG.random() > 0.5:
            plt.grid(color='k', linestyle='-', linewidth=RNG.random())

        # adjust x limit 50% of the time
        if RNG.random() > 0.5:
            plt.xlim(0)

        # Tweak spacing to prevent clipping of ylabel
        fig.tight_layout()

        # save graph to folder path
        fig.savefig(os.path.join(save_path, f'exp_{graph}.{save_format}'), dpi=save_dpi)
        plt.close(fig) # save memory usage

    # calculate runtime
    end_time = time.time()
    return f'Runtime: {round(end_time - start_time, 3)} seconds'

# %%

def norm_graph_generator(save_path:str, num_graphs=100, save_dpi=72, save_format='jpg', lorumipsum=LORUMIPSUM):
    ### generates and saves highly randomized graphs of the normal distribution ###

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

        # randomize paramters
        mu = RNG.integers(-80,80)  # mean of distribution
        sigma = RNG.integers(1,50)  # standard deviation of distribution
        x = RNG.normal(loc=mu, scale=sigma, size=RNG.integers(50,2000))
        num_bins = RNG.integers(30,120) # for histogram

        # randomize color of bins
        for i in range(3):
            color = round(RNG.random(), 1)
            rgb_bin.append(color)

        # add alpha 0 to hide bins 50% of the time (only function line is visible)
        if RNG.random() > 0.5:
            rgb_bin.append(0)

        # the histogram of the data
        count, bins, patches = ax.hist(x, num_bins, density=True, color=rgb_bin)

        # add a 'best fit' line
        y = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2)

        # color of line
        for i in range(3):
            color = round(RNG.random(), 1)
            rgb_line.append(color)

        # solid or dashed line
        if RNG.random() > 0.5:
            line_style = '-'
        else:
            line_style = '--'

        # plot line
        ax.plot(bins, y, line_style, color=rgb_line, linewidth=RNG.random()*3)
    
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

        # add grid 50% of the time
        if RNG.random() > 0.5:
            plt.grid(color='k', linestyle='-', linewidth=RNG.random())

        # Tweak spacing to prevent clipping of ylabel
        fig.tight_layout()

        # save graph to folder path
        fig.savefig(os.path.join(save_path, f'norm_{graph}.{save_format}'), dpi=save_dpi)
        plt.close(fig) # save memory usage
    # calculate runtime
    end_time = time.time()
    return f'Runtime: {round(end_time - start_time, 3)} seconds'

# %%

def lognorm_graph_generator(save_path:str, num_graphs=100, save_dpi=72, save_format='jpg', lorumipsum=LORUMIPSUM):
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
        fig.savefig(os.path.join(save_path, f'lognorm_{graph}.{save_format}'), dpi=save_dpi)
        plt.close(fig) # save memory usage

    end_time = time.time()
    return f'Runtime: {round(end_time - start_time, 3)} seconds'

# %%

if __name__ == "__main__":

    start_time = time.time()

    num_graphs = 2000

    runtime_msg = unif_graph_generator("C:\\Users\\pat_h\\htw_berlin_datasets\\dist_datasets\\unif", num_graphs)
    print(runtime_msg)
    # Runtime: 270.339 seconds (2000 graphs)

    runtime_msg = exp_graph_generator("C:\\Users\\pat_h\\htw_berlin_datasets\\dist_datasets\\exp", num_graphs)
    print(runtime_msg)
    # Runtime: 269.257 seconds (2000 graphs)
    
    runtime_msg = norm_graph_generator("C:\\Users\\pat_h\\htw_berlin_datasets\\dist_datasets\\norm", num_graphs)
    print(runtime_msg)
    # Runtime: 330.303 seconds (2000 graphs)

    runtime_msg = lognorm_graph_generator("C:\\Users\\pat_h\\htw_berlin_datasets\\dist_datasets\\lognorm", num_graphs)
    print(runtime_msg)
    # Runtime: 290.504 seconds (2000 graphs)

    end_time = time.time()
    print(f'Total Runtime: {round(end_time - start_time, 3)} seconds')
    # Total Runtime: 1160.424 seconds (2000 graphs)
    # ~20 minutes to generate 8000 graphs


# %%
