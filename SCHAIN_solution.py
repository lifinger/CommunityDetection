#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 16:12:38 2017

@author: yuql216
"""

import pandas as pd
import numpy as np
import scipy
import os
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import hdbscan

class Schain:
    def __init__(self,SA_path,SL_path,save_path,alpha = 0.5,k = 20):
        self.alpha = alpha # the coefficient of S = alpha*S1 + (1-alpha)*S2
        self.k = k         # the reduced dimension
        self.SA_path = SA_path
        self.SL_path = SL_path
        self.save_path = save_path
        self.S = self.__getS()


    def __getS(self):
        try:
            S = np.load(self.save_path+"/S.npy")
            SA = pd.read_hdf(self.SA_path,key="mydata")
            user = list(SA.index)
            self.user = user
        except:
            SA = pd.read_hdf(self.SA_path,key="mydata")
            SL = pd.read_hdf(self.SL_path,key="mydata")
            user = list(SA.index)
            self.user = user
            SL = SL.loc[user]  # row operation
            SL = SL[user]  # column operation

            SA = SA.as_matrix()
            SL = SL.as_matrix()
            SL[np.where(SL<0)] = 0.0 # 截断小于0的

            S = self.alpha * SA + (1 - self.alpha) * SL
            np.save(self.save_path+'/S.npy', S)
        return S

    def generate_U(self):
        # the process to generate reduced-dimension feature for each user
        S = self.__getS()
        try:
            D = np.load(self.save_path+"/D_minus_half.npy")
        except:
            D_ = np.sum(S, axis=1)
            D = np.diag(D_**(-0.5))
            if(np.isnan(D).any()):
                raise "nan value exist in D"
            np.save(self.save_path+'/D_minus_half.npy', D)

        try:
            K = np.load(self.save_path+"/K.npy")
        except:
            K = np.dot(D, S)
            K = np.dot(K, D)
            np.save(self.save_path+'/K.npy', K)

        try:
            Z = np.load(self.save_path+"/Z.npy")
        except:
            # Z 是一个近似解，从相同的K每次算得到的Z会略有不同，这会导致后面的U_normed不同
            # 导致最终的分群结果不一样。
            eigenvals, Z = scipy.linalg.eigh(K, eigvals=(len(self.user) - self.k, len(self.user) - 1))
            # see http://techqa.info/programming/question/12167654/fastest-way-to-compute-k-largest-eigenvalues-and-corresponding-eigenvectors-with-numpy
            np.save(self.save_path+'/Z.npy',Z)

        try:
            U_normed = np.load(self.save_path+"/U_normed.npy")
        except:
            U = np.dot(D, Z)
            # normalize U by column
            U_normed = (U - U.min(axis=0)) / (U.max(axis=0) - U.min(axis=0))
            # normalize U by row
            length = len(U_normed)
            U_normed = (U_normed - U_normed.min(axis=1).reshape(length,1) )/(U_normed.max(axis=1) - U_normed.min(axis=1)).reshape(length,1)
            np.save(self.save_path+"/U_normed.npy", U_normed)
        return U_normed

    def generate_tdid_with_label(self):
        # 利用hdbscan产生社群划分结果，只保留有社群划分的tdid
        # 保存文件为tdid_with_group
        U_normed = self.generate_U()
        U_normed = np.nan_to_num(U_normed)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=20, prediction_data=True).fit(U_normed)
        # see http://hdbscan.readthedocs.io/en/latest/soft_clustering.html
        result = clusterer.labels_

        if "tdid_with_group.csv" in os.listdir(self.save_path):
            df = pd.read_csv(self.save_path+"/tdid_with_group.csv")
        else:
            index_group = np.argwhere(result != -1)
            index_group = index_group.reshape(1, len(index_group))[0]
            tdid = [self.user[i] for i in index_group]
            group = [result[i] for i in index_group]
            df = pd.DataFrame({"tdid": tdid, "group": group})
            df.to_csv(self.save_path+"/tdid_with_group.csv", index=False)
        return result, df

if __name__ == "__main__":
    SA_path = "./results2/SA.hdf"
    SL_path = "./results2/SL.hdf"
    save_path = "./results2/"
    handle = Schain(SA_path=SA_path,SL_path=SL_path,save_path=save_path)
    result,df = handle.generate_tdid_with_label()