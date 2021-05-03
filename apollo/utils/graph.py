import pandas as pd
import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import statistics
import louvain
import igraph as ig
import leidenalg as la
from collections import Counter
from py3plex.visualization import colors
from py3plex.core import multinet
from py3plex.algorithms.community_detection import community_wrapper as cw
from tqdm import tqdm
tqdm.pandas()

sns.set(rc={'figure.figsize':(20,15)})

class Graph(object):

    def __init__(self, data_array=[]):
        self.data_array = data_array
        self.data_num = data_array.len()
        self.graphs = []
    
    def create_graphs(self):
    
    def 
    
    





if __name__ == "__main__":
    n = Graph('.\\data\\news.csv')
    #[ner,links]=n.processfile()
  
    #n.print_to_csv(links,"df_links.csv")
    #n.print_to_csv(ner,"df_ner.csv")