# %%

import numpy as np
import re

# %%

# get lorem ipsum text file
with open('./lorem_ipsum/lorem_ipsum_raw.txt') as f:
    rawlines = f.readlines()

type(rawlines)

# %%

# split the words and create unique list using set()
lines = re.split(r'[\W]', str(rawlines).lower())
setlines = set(lines)
plines = list(setlines)

# pop blank space element
plines.pop(0)
lorumipsum = plines


# %%

# test to get three random words
word_test = np.random.choice(plines, size = 3, replace=False)
# print(' '.join(word_test))
