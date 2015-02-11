"""
Fisher's linear discriminant

idea:
(1) minimize class variance : min w'(Sigma1 + Sigma2)w
(2) maximize distance between classes' mean: max ||w'miu1 - w'miu2||

"""
#author = Alison Wang(yiyin)

import numpy as np
from numpy.linalg import inv
import pandas as pd
from matplotlib import pyplot as plt
from numpy.linalg import lstsq
class FDA(object):
    def __init__(self):
        pass

    def _mean(self,X,y):
        """
        :return: mean_mat in [n_class,n_features]
        """

        classes = np.unique(y)
        n_class = classes.size

        mean_mat = []
        for cls in range(n_class):
            mean_c = np.mean(X[y==cls],axis=0)
            mean_mat.append(mean_c)

        mean_mat = np.asarray(mean_mat)
        print "Mean vector shape: ", mean_mat.shape
        return mean_mat

    def _cov(self,X,y):
        classes = np.unique(y)
        n_feature = X.shape[1]

        cov = np.zeros((n_feature,n_feature))
        cov2 = np.zeros((n_feature,n_feature))
        for idx, cls in enumerate(classes):
            print " Calculate covariance for class ", cls
            X_class = X[y==cls]
            mean_class = self.means[idx]
            cov_class = np.zeros((n_feature,n_feature))
            for row in X_class:
                # Make sure they are column vectors
                row = row.reshape((n_feature,1))
                mean_class = mean_class.reshape(n_feature,1)
                row_centered = row - mean_class
                cov_class += row_centered.dot(row_centered.T)


            cov += cov_class / X_class.shape[0]

        print cov
        return cov

    def _Sw(self,X,y, means):
        classes = np.unique(y)
        n_class = classes.size

        n_feature = X.shape[1]
        within_mat =np.zeros((n_feature,n_feature))
        for cls in range(n_class):
            print "Class: ", cls
            #print means[cls]
            #print X[y==cls,][0:5]

            cls_sc_mat = np.zeros((n_feature,n_feature))
            for row in X[y==cls]:
                delta = row - means[cls]
                #print delta
                cls_sc_mat += delta.dot(delta.T)
            within_mat += cls_sc_mat

        #within_mat = np.asarray(within_mat)
        print "Scatter matrix: ", within_mat.shape
        print within_mat
        return within_mat

    def _weight(self,X, cov, means):
        """
        By solving max[ w'Sbw / w'Sw w] by lagrandian,
        inv(Sw)Sb w = lamda w
        That is, weight is the largest eigenvector of Sw.inv Sb
        The eigenvector will construct new space where distance between classes' means are maximized and variance between classes are minimized
        returns weight vector n_feature
        """
        #weight = inv(cov).dot(means)

        # Solves the equation cov x = means by computing a vector x that minimizes the Euclidean 2-norm || mean - cov x ||^2.
        # when inverse of covariance exist, the answer is equivalent to inv(Cov).dot(means.T)
        # Each class mean should be a column.
        weight = lstsq(cov, means.T)[0]

        print weight
        print inv(cov).dot(means.T)
        print "Different approach for weight"

        return weight

    def _intercept(self, cov, means, priors):

        weight = np.dot(cov,means)
        intercept = -0.5 * np.diag(np.dot(means, weight))+ np.log(priors)
        return intercept

    def fit(self,X,y):

        m,n = X.shape
        classes = np.unique(y)
        n_class = classes.size

        p1 = np.mean(y)
        priors = np.array([1-p1,p1])

        means = self._mean(X,y)
        self.means = means

        cov = self._cov(X,y)

        #Sw = self._Sw(X,y,means)

        weight = self._weight(X, cov,means)
        intercept = self._intercept(cov,means,priors)

        self.priors = priors
        self.n_class = n_class
        self.classes = classes
        self.m_sample = m
        self.n_feature = n
        self.weight = weight
        self.intercept = intercept
        return self

    def decision_function(self,X):
        projection = X.dot(self.weight) + (self.intercept)

        # if projection > 0, assign to class 0
        return projection

    def predict(self,X):
        pred = self.decision_function(X)
        print "Pred shape", pred.shape
        return self.classes[pred.argmax(1)]

    def predict_prob(self,X):
        proj = self.decision_function(X)

        # Base on Gaussian assumption
        likelihood = np.exp(proj - proj.max(axis=1)[:,np.newaxis])
        # print proj[0:5,:]
        # print proj.max(axis=1)[:,np.newaxis][0:5]
        # print (proj - proj.max(axis=1)[:,np.newaxis])[0:5,:]
        return likelihood / likelihood.sum(axis=1)[:, np.newaxis]

# Use Iris dataset for test
df =pd.read_csv("data/D2_train.csv")
print df.info()
print df.head()

X = np.asarray(df.ix[:,:-1])
y = np.asarray(df['y'])

print "Data shape: ", X.shape
print "Label shape: ", y.shape

fda = FDA()
fda.fit(X,y)
prediction = fda.predict(X)
print "Accurancy: ", np.mean(0+(prediction==y))


# Plot prediction
def plotLDA(X,y,clf,color):
    # create a mesh to plot in

    h = .2 # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

    # concatenate xx and yy
    z = clf.predict_prob(np.c_[xx.ravel(), yy.ravel()])[:,1]
    print z.shape
    z = z.reshape(xx.shape)

    plt.contour(xx,yy,z,
                [0.5], # threshold
                colors = color)


def plot(X,y, dec_boundary=None):
    plt.clf()

    # plot data points
    colors =['red','blue']
    for label in np.unique(y):
        plt.scatter(X[:,0][y==label], X[:,1][y==label], color=colors[label])

    plotLDA(X,y,fda,"black")
    plt.show()


plot(X,y)