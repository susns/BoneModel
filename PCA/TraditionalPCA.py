import numpy as np
from sklearn.decomposition import PCA
from tools import distance_RMSE
import ssmfit as ssmfit


class TraditionalPCA:
    def __init__(self, X, scale=None):
        """
        :param X: shape (b,3n)
        :param scale: (b,)
        """
        self.X = X
        self.scale = scale
        self.N = min(X.shape)
        self.pca = PCA(n_components=self.N)

        # 利用数据训练模型（即上述得出特征向量的过程）
        self.pca.fit(X)
        # 得出原始数据的降维后的结果；也可以以新的数据作为参数，得到降维结果。
        self.b = self.pca.transform(X)
        self.p = self.pca.components_
        self.m = self.pca.mean_

    def print_counts(self, n):
        if n > self.N:
            print('Too larger!')
            return

        b = self.b.copy()
        b[:, n:] = 0

        Y = self.m + np.dot(b, self.p)
        dis = distance_RMSE(self.X, Y, self.scale)

        # 打印各主成分的方差占比
        # print("dimension {}/{}".format(n, self.X.shape[1]))
        # print('ratio', self.pca.explained_variance_ratio_[:n], np.sum(self.pca.explained_variance_ratio_[:n]))
        # print('variance', self.pca.explained_variance_[:n])
        # print('mean distance: {0:.4f}'.format(dis))
        return np.sum(self.pca.explained_variance_ratio_[:n]), dis

    def get_one(self, k, v):
        k = k-1
        ab = np.zeros(self.N).reshape(1, self.N)
        ab[0][k] = v*np.sqrt(self.pca.explained_variance_[k])
        aY = self.m + np.dot(ab, self.p)
        return aY

    def get_pre(self, k, v):
        ab = np.zeros(self.N).reshape(1, self.N)
        ab[0, :k] = v*np.sqrt(self.pca.explained_variance_[:k])
        aY = self.m + np.dot(ab, self.p)
        return aY

    def fit(self, V, nmodes):
        '''
        Args:
            V: (M, 3)
            nmodes: the number of primary components

        Returns:
            SSMfit: (M, 3)

        '''
        MEAN = self.m.reshape(-1,3)
        ssmV = ssmfit.change_ssmV(self.p)
        # ReallignedV 是将V与SSM进行刚性配准后的坐标
        # SSMfit是拟合效果
        RMSerror, ReallignedV, transform, SSMfit, EstimatedModes, position_error = ssmfit.SSMfitter(MEAN, ssmV, V, nmodes)
        return SSMfit, ReallignedV