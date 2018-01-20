#--coding:utf-8--
import pandas as pd
from pandas import *
import graphlab as gl
import os
import numpy as np
from scipy.spatial import distance

class node_att:
    def __init__(self,graph_path,bq_path):
        self.graph, self.bq = self.__load_file(graph_path,bq_path)
        self.gp_bq = self.getGp()
        # self.similarity = self.getPairWiseSim()

    def __load_file(self,graph_path,bq_path):
        print("loading file...")
        graph = pd.read_csv(graph_path, delimiter="\t", header=None,
                            names=["bssid", "tdid", "tdid_index", "weight", "lat", "lng"])
        bq = pd.read_csv(bq_path, delimiter="\t", header=None,
                         names=["tdid", "tdid2", "label_index", "label", "weight"])
        del (bq["tdid2"])
        # only reserve the bottom level tag, which means the index are bigger than 1000000
        bq = bq[bq["label_index"] > 1000000]
        bq["dic"] = bq.apply(self.__return_dic, axis=1)
        return graph, bq

    def getGp(self):
        print ("get tfidf for each user...")
        gp_bq_0 = self.bq.groupby("tdid").apply(self.__combine_dic).reset_index(drop=False)
        gp_bq = gl.SFrame(gp_bq_0)
        gp_bq["tfidf"] = gl.text_analytics.tf_idf(gp_bq["dic"])
        return gp_bq

    def getPairWiseSim(self):
        print ("calculate the label dict...")
        # 生成label dictionary
        label_dict = {}
        label = self.bq["label"]
        label = label.drop_duplicates()
        for i, label in enumerate(label.values):
            label_dict[label] = i

        print ("calculate similarity matrix...")
        tfidf = self.__tfidf_matrix(self.gp_bq["tfidf"], label_dict)
        np.save("./results2/tfidf.npy",tfidf)
        SA = self.__cosine_matrix(tfidf, self.gp_bq)
        SA.to_hdf("./results2/SA.hdf",key="mydata")
        return SA

    def __tfidf_matrix(self,tfidf, label_dict):
        # 将所有用户的tfidf转换为一个n*203的矩阵
        m = np.zeros((len(tfidf), len(label_dict)))
        for i in range(len(tfidf)):
            d = tfidf[i]
            for k, v in d.items():
                m[i, label_dict[k]] = v
        return m

    def __cosine_matrix(self,m, gp_bq):
        # 根据tfidf的矩阵，计算出pair wise的cosine距离
        # 因为tfidf的值都是正数，所以不需要利用0.5做归一化，而link similarity需要
        cosine_matrix = 1 - distance.cdist(m, m, "cosine")
        SA = pd.DataFrame(cosine_matrix, index=gp_bq["tdid"], columns=gp_bq["tdid"])
        return SA

    def __return_dic(self,row):
        # combine the label and weight of each row to form a dict
        if long(row["weight"]) > 0:
            return {row["label"]: long(row["weight"])}
        else:
            return {row["label"]: 1L}

    def __combine_dic(self,group):
        # combine the dict of each group to a union dict
        dic = {}
        for d in group["dic"].values:
            dic.update(d)
        return Series({"dic": dic})


if __name__ =="__main__":
    graph_path = "./data/miniGraph"
    bq_path = "./data/biaoqian_dongcheng_tdid_10000"
    handle = node_att(graph_path,bq_path)
