#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import pandas as pd
import networkx as nx
import numpy as np
from tqdm import trange
from sklearn import manifold
from sklearn.cluster import KMeans
from scipy.spatial import distance
import os

class link_sim:
    def __init__(self,graph_path,bq_path,embedding_n = 20):
        miniGraph = pd.DataFrame.from_csv(graph_path,sep='\t', header=None)
        miniGraph.columns = ['tdid', 'tdid_idex', 'weight', 'lat', 'long']
        graph = miniGraph[['tdid', 'weight']]
        self.graph = graph
        self.bq = pd.DataFrame.from_csv(bq_path,sep="\t",header=None)
        self.user = self.bq.index.unique()
        self.embedding_n = embedding_n # the dimensions of embedding after spectral embedding
        self.G = self.indexedGraph()

    def indexedGraph(self):
        #  return Graph
        weighted_edge_list = [(index, row[0], row[1]) for index, row in self.graph.iterrows()]
        G = nx.Graph()
        G.add_weighted_edges_from(weighted_edge_list)
        self.G = G
        return G

    def embed(self,save=False):
        try:
            embedding = np.load("./results2/embedding.npy")
        except:
            G = self.indexedGraph()
            "starting manifold learning..."
            A = nx.adjacency_matrix(G, nodelist=G.nodes())
            embedding = manifold.spectral_embedding(A, n_components=self.embedding_n)
            if save:
                np.save("./results2/embedding.npy",embedding)
        self.embedding = embedding
        return embedding

    def getPairWiseSim(self):
        embedding = self.embed(save=True)
        cosine_matrix = (1 - distance.cdist(embedding, embedding, 'cosine'))
        cosine_matrix[np.where(cosine_matrix<0)] = 0.0
        SL = pd.DataFrame(cosine_matrix, index=self.G.nodes(), columns=self.G.nodes())
        ## filter out wifi nodes
        SL = SL.loc[self.user]
        SL = SL[self.user]
        return SL

if __name__ == '__main__':
    graph_path = "./data/miniGraph"
    bq_path = "./data/biaoqian_dongcheng_tdid_10000"
    handle = link_sim(graph_path,bq_path)
    SL= handle.getPairWiseSim()
    SL.to_hdf('./results2/SL.hdf', key='mydata')


