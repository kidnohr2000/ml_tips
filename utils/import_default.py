import os
import re
import multiprocessing
import subprocess
import concurrent.futures as fu
import gc

import pandas as pd
import pandas_profiling as pdp
import numpy as np
from tqdm import tqdm_notebook as tqdm

from IPython.display import display, HTML
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st


import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 200)
# カラム内の文字数
pd.set_option("display.max_colwidth", 200)
#行数
pd.set_option("display.max_rows", 200)

from matplotlib import rcParams
rcParams['font.family'] = 'IPAPGothic'

%matplotlib inline

# from IPython.core.debugger import Pdb; Pdb().set_trace()


def cv2_imshow(title, image):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    # as opencv loads in BGR format by default, we want to show it in RGB.
    plt.show()
