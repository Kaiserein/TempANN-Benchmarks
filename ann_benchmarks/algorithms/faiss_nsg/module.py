import faiss
import numpy as np

from ..faiss.module import Faiss

class FaissNSG(Faiss):
    def __init__(self, metric, R):
        self.metric = metric
        self.R = R
        
    def fit(self, X):   #X是向量矩阵
        self.index = faiss.IndexNSGFlat(len(X[0]), self.method_param["R"])
        
        self.index.verbose = True   #启用 Faiss 索引的详细输出
        
        if self._metric == "angular":   #把angular度量方式换成L2
            X = X / np.linalg.norm(X, axis=1)[:, np.newaxis]    #如果指定的度量是“angular”（角度距离），则将输入数据 X 规范化。
        if X.dtype != np.float32:   #确保了输入数据 X 是 float32 类型。
            X = X.astype(np.float32)

        self.index.add(X)
        faiss.omp_set_num_threads(1)    #设置 OpenMP 的线程数目为 1
        
    def set_query_arguments(self, ef):
        faiss.cvar.nsg_stats.reset()
        self.index.nsg.efSearch = ef

    def get_additional(self):
        return {"dist_comps": faiss.cvar.nsg_stats.ndis}

    def __str__(self):
        return "faiss (%s, ef: %d)" % (self.method_param, self.index.nsg.efSearch)
    
    def freeIndex(self):
        del self.p