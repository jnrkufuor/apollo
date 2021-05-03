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
        self.graphs = []
        self.unique_nodes = []
    
    def create_graphs(self):
        ''' Function to create graphs from self.graphs(can be more than one graph) and stores the graphs inside the self.graphs 
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
            self.graphs.append(graph)
            print(graph)
            print("Graph created ...")
        
    def get_graphs(self):
        ''' Function to return created graphs

            :return grahs: List of graphs
        '''   
        if not self.graphs:
            print("No graphs created. Create Graphs first")
            return
        else:
            return self.graphs
        
        
    def pull_unique_nodes(self):
        ''' Function to extract unique nodes from data

            :return unique_nodes: List of unique nodees
        '''
        if not self.graphs:
            print("No graphs created. Create Graphs first")
            return
        unique_nodes=[]
        for row in self.graphs[0].iterrows():
            if row[1]["from"] not in unique_nodes: unique_nodes.append(row[1]["from"])
            if row[1]["to"] not in unique_nodes: unique_nodes.append(row[1]["to"])
        return unique_nodes
    
    
    def visualize_graphs(self):
        ''' Function to visualize graphs using the Kamada Kawai Layout
        '''
        for G in self.graphs:
            pos = nx.kamada_kawai_layout(G)
            nodes = G.nodes()
            fig, axs = plt.subplots(1, 1, figsize=(15,20))
            el = nx.draw_networkx_edges(G, pos, alpha=0.1, ax=axs)
            nl = nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color='#FF427b', 
                                        node_size=50, ax=axs)
            ll = nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
            plt.show()
        
    def generate_graph_labels(self):
        ''' Function to generate various labels and nodes for the multilayer graph use
        
            :return :
        '''
        node_labels = {} 
        nodes_multi_layer={}
        type_count=0
        node_count =0
        for G in self.graphs:
            node_type ="t"+(type_count+1)
            for node in G.nodes():
                #set the node name as the key and the label as its value 
                node_labels[node] = node
                
                #create nodes for multilayered graph
                nodes_multi_layer[node_count]={"node": node,"type":node_type}

                node_count+=1
            type_count+=1
        return [node_labels,nodes_multi_layer]
    
    def get_centralities(self):
        ''' Function to generate various labels and nodes for the multilayer graph use
        
            :return :
        '''
        centralities =[]
        for G in self.graphs:
            nodes = []
            eigenvector_cents = []
            ec_dict = nx.eigenvector_centrality(G, max_iter=1000, weight='weight')
            for node in tqdm(G.nodes()):
                nodes.append(node)
                eigenvector_cents.append(ec_dict[node])

            df_centralities = pd.DataFrame(data={'entity': nodes,
                                                'eigenvector': eigenvector_cents})

            centralities.append(df_centralities)
        return centralities

    def visualize_centralities_barplot(self,centralities,filter=20, save_figure=False):
        ''' Function to generate various labels and nodes for the multilayer graph use
            :param centralities: list of centralities of each graph
            :param filter: number top records to filter on
            :param save_figure: boolean to export the image or not
        '''
        count = 1
        for centrality in centralities:
            df_cent_top = centrality.sort_values('eigenvector', ascending=False).head(filter)
            df_cent_top.reset_index(inplace=True, drop=True)
            fig, axs = plt.subplots(figsize=(10,7))
            g = sns.barplot(data=df_cent_top,
                        x='eigenvector',
                        y='entity',
                        dodge=False,
                        orient='h',
                        hue='eigenvector',
                        palette='viridis',)

            g.set_yticks([])
            g.set_title('Most influential entities in network Graph '+count )##
            g.set_xlabel('Eigenvector centrality')
            g.set_ylabel('')
            g.set_xlim(0, max(df_cent_top['eigenvector'])+0.1)
            g.legend_.remove()
            g.tick_params(labelsize=5)

            for i in df_cent_top.index:
                g.text(df_cent_top.iloc[i]['eigenvector']+0.005, i+0.25, df_cent_top.iloc[i]['entity'])

            ut.save_figure(g,'cent_plot.png')
            nodes = []
            eigenvector_cents=[]
            count+=1
        
        
            

        
        
    





if __name__ == "__main__":
    n = Graph('.\\data\\news.csv')
    #[ner,links]=n.processfile()
  
    #n.print_to_csv(links,"df_links.csv")
    #n.print_to_csv(ner,"df_ner.csv")