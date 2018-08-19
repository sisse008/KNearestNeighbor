import numpy as np
import heapq as hq

class NearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, Y):
        """   X is N x D where each row is an example. Y is 1- dimension of size N """

        self.Xtr = X
        self.Ytr = Y

    def predictL1(self, X):
        """ X is an array of images we want to predict a label for. each row is an image. an image is just an array of 32 * 32 * 3 pixels (pixel value 0-255)"""
        # X.shape[0] = number of images
        # X.shape[1] = dimension of an image (32 * 32* 3)
        num_test = X.shape [0]

        Ypred = np.zeros(num_test, dtype = self.Ytr.dtype)

        #for each image
        for i in xrange(num_test):
            # using L1 distance from each training data image to test image
            # shape of distances = N
            # self.Xtr - X[i,:] = array of differences (each row in self.Xtr - row i in X) shape = N * D
            # np.abs(self.Xtr - X[i,:]) = turns all negative values to positive
            # np.sum(np.abs(self.Xtr - X[i,:]), axis = 1) = return an array holding the sum of each row

            distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)

            # find the smallest distnace for this image
            min_index = np.argmin(distances)
            # add prediction to prediction array
            Ypred[i] = self.Ytr[min_index]

        return Ypred

    def predictL2(self, X):
        """ X is an array of images we want to predict a label for. each row is an image. an image is just an array of 32 * 32 * 3 pixels (pixel value 0-255)"""
        # X.shape[0] = number of images
        # X.shape[1] = dimension of an image (32 * 32* 3)
        num_test = X.shape [0]

        Ypred = np.zeros(num_test, dtype = self.Ytr.dtype)

        #for each image
        for i in xrange(num_test):

            distances = np.sqrt(np.sum(np.square(self.Xtr - X[i, :]), axis=1))

            # find the smallest distnace for this image
            min_index = np.argmin(distances)
            # add prediction to prediction array
            Ypred[i] = self.Ytr[min_index]

        return Ypred


    def KpredictL2(self, X, k):
        """ X is an array of images we want to predict a label for. each row is an image. an image is just an array of 32 * 32 * 3 pixels (pixel value 0-255)"""
        # X.shape[0] = number of images
        # X.shape[1] = dimension of an image (32 * 32* 3)
        num_test = X.shape [0]

        Ypred = np.zeros(num_test, dtype = self.Ytr.dtype)

        #for each image
        for i in xrange(num_test):

            distances = np.sqrt(np.sum(np.square(self.Xtr - X[i, :]), axis=1))

            # find the k smallest distnaces for this image
            k_min_indexes = hq.nsmallest(k, range(len(distances)), distances.take)

            #find highest vote
            votes =

            # add prediction to prediction array
            Ypred[i] = self.Ytr[min_index]
        C:\Users\Danielle
        Sisserman\AppData\Local\Programs\Python\Python36 - 32\DLLs\NearestNeighbor.py
        return Ypred