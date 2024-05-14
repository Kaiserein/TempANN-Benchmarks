import faiss
import numpy as np

from ..faiss.module import Faiss


class FaissHNSW(Faiss):
    def __init__(self, metric, method_param):
        self._metric = metric
        self.method_param = method_param

    def fit(self, X):   
        self.index = faiss.IndexHNSWFlat(len(X[0]), self.method_param["M"])
        self.index.hnsw.efConstruction = self.method_param["efConstruction"]
        self.index.verbose = True

        if self._metric == "angular":   
            X = X / np.linalg.norm(X, axis=1)[:, np.newaxis]
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        self.index.add(X)
        faiss.omp_set_num_threads(1)

    def set_query_arguments(self, ef): #用于设置搜索查询时使用的参数。ef参数表示搜索时使用的候选项数目。
        faiss.cvar.hnsw_stats.reset() #重置HNSW统计信息。
        self.index.hnsw.efSearch = ef #设置搜索时使用的候选项数目为ef。

    def get_additional(self): #定义了一个get_additional方法，用于获取一些额外的信息，如距离计算的次数。
        return {"dist_comps": faiss.cvar.hnsw_stats.ndis} #返回一个字典，包含距离计算的次数。
 
    def __str__(self): #定义了类的字符串表示方法，返回一个字符串，描述了当前的索引类型和搜索时使用的候选项数目。
        return "faiss (%s, ef: %d)" % (self.method_param, self.index.hnsw.efSearch)

    def freeIndex(self): #用于释放索引资源。
        del self.p