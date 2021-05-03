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
from apollo_util import Util
from py3plex.visualization import colors
from py3plex.core import multinet
from py3plex.algorithms.community_detection import community_wrapper as cw
from tqdm import tqdm
tqdm.pandas()
ut = Util()

sns.set(rc={'figure.figsize':(20,15)})

class Graph(object):

    def __init__(self, data_array=[pd.DataFrame()]):
        ''' Initialization function for named entity recognition parts

            :param path_to_data: Path to news content
        '''
        self.data_array = data_array
        self.data_num = data_array.len()
        self.graphs = pd.DataFrame()
        self.unique_nodes = []
    
    def create_graphs(self):
        ''' Initialization function for named entity recognition parts

            :param path_to_data: Path to news content
        '''
        for i in range(self.data_num):
            self.data_array[i].reset_index(inplace=True, drop=True)
            graph = nx.Graph()
            if not self.unique_nodes:
                self.unique_nodes = self.pull_unique_nodes()
            for nd in self.unique_nodes:
                graph.add_node(nd)
            for link in tqdm(self.data_array[i].index):
                graph.add_edge(self.data_array[i].iloc[link]['from'],self.data_array[i].iloc[link]['to'],weight=self.data_array[i].iloc[link]['weight'])
                
            
    
    
    def pull_unique_nodes(self):
        ''' Initialization function for named entity recognition parts

            :param path_to_data: Path to news content
        '''
        unique_nodes=[]
        for row in self.graphs[0].iterrows():
            if row[1]["from"] not in unique_nodes: unique_nodes.append(row[1]["from"])
            if row[1]["to"] not in unique_nodes: unique_nodes.append(row[1]["to"])
        return unique_nodes
        
        

    
    
    





if __name__ == "__main__":
    n = Graph('.\\data\\news.csv')
    #[ner,links]=n.processfile()
  
    #n.print_to_csv(links,"df_links.csv")
    #n.print_to_csv(ner,"df_ner.csv")