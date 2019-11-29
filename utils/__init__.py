#import MySQLdb
import pandas as pd
from contextlib import closing
import matplotlib.pyplot as plt
import os
import time
import math
from datetime import datetime
from tqdm import tqdm
import numpy as np
import itertools
import subprocess
import pickle as pkl
from dateutil.relativedelta import relativedelta
from multiprocessing import Process,Queue,Pool,Manager
import codecs
#plt.style.use('ggplot')

#from .sql_functions import *