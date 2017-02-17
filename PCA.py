import numpy as np

# For general PCA
# @author pknam
class PCA:
    # dim : dimesion of data
    def __init__(self, dim):
        self.dim = 0
        self.data = None
        self.eigVal = None
        self.eigVec = None

    # data : length is dim
    def addData(self, data):
        if self.data is None:
            self.dim = data.size
            self.data = data.reshape((self.dim, 1))
            return

        # for hstack
        data = data.reshape((self.dim, 1))
        self.data = np.hstack((self.data, data))

    def setData(self, data):
        self.data = data

    # calc mean data
    def mean(self):
        return self.data.mean(axis=1)

    def doPCA(self):
        covMat = np.cov(self.data)
        eigVal, eigVec = np.linalg.eigh(covMat)

        # reverse so that 0-index be the largest
        # eigVal[i] and eigVec[i] are a pair
        self.eigVal = eigVal[::-1]
        self.eigVec = eigVec[:,::-1].T

    # reconstruct data with k-largest principle vector
    def reconstruct(self, data, k):
        if self.dim < k:
            k = self.dim

        reconstructedData = np.zeros((self.dim))

        # KLT (Karhunen loeve transform)
        for i in xrange(k):
            coeff = np.dot(data, self.eigVec[i])
            basis = self.eigVec[i]
            reconstructedData += coeff * basis

        return reconstructedData