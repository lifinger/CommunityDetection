#--coding:utf-8--
import numpy
import pandas as pd
from pandas import *
import hdbscan
import numpy as np
import graphlab as gl
import os
import numpy as np
from scipy.spatial import distance
from matplotlib import pyplot as plt
import matplotlib

'''分析社群划分以后的各个群组里面的tfidf情况'''
class group_tfidf:
    def __init__(self,group_path,file_bq):
        self.tdid_group = pd.read_csv(group_path)
        self.bq = pd.read_csv(file_bq, delimiter="\t",header = None,\
                              names=["tdid", "tdid2", "label_index", "label", "weight"])
        del(self.bq["tdid2"])
        self.bq = self.bq[self.bq["label_index"] > 1000000]

        self.table = pd.merge(self.bq, self.tdid_group, on="tdid")
        self.table["dic"] = self.table.apply(self.return_dic, axis=1)

        gp_table_0 = self.table.groupby("group").apply(self.combine_dic).reset_index(drop=False)
        gp_table_1 = self.table.groupby("group").apply(self.getSize).reset_index(drop=False)
        # self.gp_table = pd.concat([gp_table_0,gp_table_1],axis=1)
        self.gp_table = pd.merge(gp_table_0,gp_table_1,on="group")
        self.gp_table = gl.SFrame(self.gp_table)
        self.gp_table["tfidf"] = gl.text_analytics.tf_idf(self.gp_table["dic"])

        # 得到分社群的size，tfidf信息后，可以进一步提取排名靠前的tfidf 信息
        self.gp_table["TOP-5-TFIDF"] = self.gp_table.apply(self.getTopTFIDF)

    def return_dic(self,row):
        # combine the label and weight of each row to form a dict
        if long(row["weight"]) > 0:
            return {row["label"]: long(row["weight"])}
        else:
            return {row["label"]: 1L}

    def combine_dic(self,group):
        # combine the dict of each group to a union dict
        dic = {}
        for d in group["dic"].values:
            dic.update(d)
        return Series({"dic": dic})

    def getSize(self,group):
        tdid_unique = group["tdid"].drop_duplicates()
        size = len(tdid_unique)
        return Series({"size":size})

    def getTopTFIDF(self,row):
        items = row["tfidf"].items()
        items = sorted(items,key=lambda x : -x[1])
        if len(items)>5:
            return items[:5]
        else:
            return items

'''对每个分组的社群画出排名靠前五的TFIDF'''
def generate_TOP_tfidf_figures(group_path,bq_path,save_path):
    g = group_tfidf(group_path,bq_path)
    df = g.gp_table.to_dataframe()
    # zhfont1 = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\MSYH.ttc')
    def draw_bar(group_index,row,save_path):
        tfidf = row["TOP-5-TFIDF"]
        quants = []
        labels = []
        for lis in tfidf:
            labels.append(lis[0])
            quants.append(lis[1])

        size = row["size"]
        width = 0.4
        ind = np.linspace(0.5,0.5*len(labels),len(labels))
        fig = plt.figure(1)
        ax = fig.add_subplot(111)

        # bar plot
        ax.bar(ind-width/2,quants,width,color="green")

        for i in range(len(labels)):
            ax.text(x=ind[i]-width/2-0.1,y=quants[i]+0.05*max(quants), s=unicode(labels[i],"utf8"), fontsize=12)
        ax.set_xlabel("label")
        ax.set_ylabel("TF-IDF value")
        ax.set_title("Group {group_index} with {num} people".format(group_index=group_index-1,num=size))
        plt.grid(True)
        plt.show()
        plt.savefig(save_path + "/group_cut_%d.png"%(group_index-1))
        plt.close()

    for index in range(len(df)):
        draw_bar(index,df.iloc[index],save_path)
    return g

if __name__ == "__main__":
    group_path = "./results2/tdid_with_group.csv"
    bq_path = "./data/biaoqian_dongcheng_tdid_10000"
    h = group_tfidf(group_path,bq_path)
    # 打印每个群组的top5 tfidf标签
    for row in h.gp_table:
        print "\n"
        print row["group"],row["size"]
        dic = reduce(lambda x,y:x+y,row["TOP-5-TFIDF"])
        for i in dic:
            print i,