import faiss
import numpy as np

from ..faiss.module import Faiss

class FaissNSG(Faiss):
    def __init__(self, metric, method_param):
        self._metric = metric
        self.method_param = method_param

    def fit(self, X):
        d = X.shape[1]  # 向量维度
        if self._metric == 'angular':   #归一化处理
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        self.index = faiss.IndexNSGFlat(d, self.method_param["R"])  # 使用 NSG 算法构建索引, R代表邻居数量
        self.index.verbose = True

        if 'K' in self.method_param: #K 最近邻图（KNN graph）中每个节点的邻居数量。默认值为 64
            self.index.K = self.method_param['K']
        if 'nndescent_S' in self.method_param: # NNDescent 算法中的 S 参数，指定了随机探测的样本数目，默认值为 10
            self.index.nndescent_S = self.method_param['nndescent_S']
        if 'nndescent_R' in self.method_param: # NNDescent 算法中的 R 参数，指定了最大近邻搜索次数，默认值为 100
            self.index.nndescent_R = self.method_param['nndescent_R']
        if 'nndescent_L' in self.method_param: # NNDescent 算法中的 L 参数，指定了最大搜索路径长度，默认值为 GK + 50
            self.index.nndescent_L = self.method_param['nndescent_L']
        if 'nndescent_iter' in self.method_param: # NNDescent 算法中的迭代次数。默认值为 10
            self.index.nndescent_iter = self.method_param['nndescent_iter']
        if 'Search_L' in self.method_param: #搜索路径长度
            self.index.Search_L = self.method_param['Search_L']
        if 'C' in self.method_param: #候选集大小
            self.index.C = self.method_param['C']
        self.index.train(X)
        self.index.add(X)
        faiss.omp_set_num_threads(1)

    def set_query_arguments(self, ef): #没有需要额外设置的参数，所有参数都在fit阶段设置完毕
        pass

    def get_additional(self): #没有额外参数，因此不需要返回任何内容
        return {}

    def __str__(self):
        return "FaissNSG"

    def freeIndex(self): #释放占用的资源
        del self.index
