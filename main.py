import os
import pandas as pd
import glob
import numpy as np

from itertools import combinations

import networkx as nx
from networkx.algorithms import community

import nltk
nltk.download('wordnet')
!pip install gensim
import pandas as pd
import  nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
import re
import matplotlib.pyplot as plt

#from google.colab import drive
#drive.mount("/content/drive/", force_remount=True)

files_path = "C:\Users\zaure\Documents\Research\comment_01"


pip install detoxify
pip install "git+https://github.com/tqdm/tqdm.git@devel#egg=tqdm"

from utils import read_files
from utils import predict_lang
from utils import preprocess_data

comments_data= read_files(files_path)
comments_data['language']= predict_lang(comments_data, 'comment_message')
comments_data['text']= comments_data['comment_message']
preprocess_data(comments_data,'text')
df_ru = comments_data[comments_data['Language']=='ru']
predict_toxicity(10)
