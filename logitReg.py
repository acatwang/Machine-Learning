"""
Logistic Regression

"""
#Author: Alison Wang(yiyin)


import numpy as np
import pandas as pd
from sklearn import preprocessing
import random

def load_train(csvfile, normalize=True, addConst=True):
    """
    :param csvfile: the path of input file
    :param normalize: Boolean. Standardize the features to have a standard normal distribution
    :param addConst: Boolean, Add ones at the beginning of X

    :return:
    X : [m_sample, n_feature] or [m_sample, n_feature+1]
    y : [m_sample] array
    """
    print "Loading train data..."
    data = pd.read_csv(csvfile)
    #print data.head()
    print data.info()
    y = data['label']
    X = data.ix[:, 1:-1] # drop id and label column

    if normalize:
        X = preprocessing.scale(X, axis=0, with_mean=True, with_std=True, copy=False)

    if addConst:
        X = np.hstack(( np.ones((X.shape[0], 1), dtype=X.dtype),X))

    print X[0:5,:]
    X = np.asarray(X)
    y = np.asarray(y)


    print ("The dimension of data: ", X.shape)
    print ("The dimension of data_y: ", y.shape)
    return X,y

def load_test(csvfile,normalize=True,addConst=True):
    """
    :param csvfile: the path of input file
    :param normalize: Boolean. Standardize the features to have a standard normal distribution
    :param addConst: Boolean, Add ones at the beginning of X

    :return:
    X : [m_sample, n_feature] or [m_sample, n_feature+1]

    """
    print "Loading test data..."
    data = pd.read_csv(csvfile)
    data.info()
    X = data.ix[:, 1:] # drop id column

    if normalize:
        X = preprocessing.scale(X, axis=0, with_mean=True, with_std=True, copy=False)

    if addConst:
        X = np.hstack((X, np.ones((X.shape[0], 1), dtype=X.dtype)))

    print X[0:5,:]
    X = np.asarray(X)

    return X

def sigmoid(z):
    return 1.0/ (1.0+np.exp(-z))

class LogitReg(object):

    def __init__(self,alpha,lamda,iterations):

        # Weight decay. Affect regulation term.
        # Smaller number means stronger regulations
        self.lamda = lamda

        # alpha: Learning rate.
        # iteration: How many times to iterate for gradient decent
        self.alpha = alpha
        self.iterations = iterations

    def _hypothesis(self,X,theta):
        """
        arg
        X : shape [n_sample, n_feature]
        theta: n_feature array

        return
        y_hat = g(theta'X), where g is a sigmoid function
        """

        z = np.dot(X,theta)
        return sigmoid(z)

    def _cost(self,theta, X,y, lamda):
        """
        Calculate the log-likelihood funciton

        :return: an number(cost)
        """
        m,n = X.shape

        h_theta = self._hypothesis(X,theta)
        # Suppose class = 2

        # Log Prob(t|w)
        cost_vec = (y * np.log(h_theta) + (1-y) * np.log(1-h_theta))

        # regularize term

        reg_term = .5 * float(lamda) * (theta**2)

        J = -sum(cost_vec)/m + sum(reg_term[1:])/m

        return J

    def _gradient(self,theta,X,y,lamda):
        """
        Calculate the L2 regularized gradient of log-likelihood function,
        given the theta,X,y and lamda

        :return: n_feature array
        """
        m,n = X.shape
        h_theta = self._hypothesis(X,theta)

        reg_term_gradient = float(lamda) * theta/m
        gradient = (h_theta - y).T.dot(X)/m

        grad = gradient + reg_term_gradient
        grad[0] -= reg_term_gradient[0]

        return grad

    def gradientDecent(self,X,y,alpha,interations):
        """

        :param X: m_sample * n_feature
        :param y: m_sample
        :param alpha: learning rate
        :param interations: max interation number

        :return: optimize theta (n_feature array)
        """
        m,n = X.shape
        theta = np.zeros(n) # initial theta

        for i in range(interations):

            cost = self._cost(theta,X,y,self.lamda)
            gradient = self._gradient(theta,X,y,self.lamda)

            print "Iteration %s: %s" % (i,cost)
            # update
            theta = theta - alpha * gradient.T


        return theta

    def fit(self,X,y):
        """

        :return:
        theta_ : for 2 classes case, return n_feature array
        thetas_: for multiple classes, return [n_feature, n_classes] matrix
        """
        m, n  = X.shape
        classes = np.unique(y)
        n_class = len(classes)

        print "Number of classes"
        print n_class

        if n_class == 2:
            theta_opt = self.gradientDecent(X,y,self.alpha,self.iterations)
            self.theta_ = theta_opt

        elif n_class >2:
            print "fitting for multiple classes case"
            # Create sparse y matrix
            y_sparse = np.zeros((m,n_class))

            rowcnt = 0
            for label in y:
                y_sparse[rowcnt][label] = 1
                #print ("label: ", label)
                #print y_sparse[rowcnt]
                rowcnt +=1

            print "Make sure sparse data is the same as y"
            print np.argmax(y_sparse,axis=1) == y


            thetas = []
            for cls in range(n_class):
                theta_opt_class = self.gradientDecent(X,y_sparse[:,cls], self.alpha,self.iterations)
                thetas.append(theta_opt_class)

            thetas = np.asarray(thetas)
            print "Theta for classes"
            print thetas.shape
            self.thetas_ = thetas.T

        return self

    def decision_function(self,X,one_v_all=False):
        """
        Assign labels based on X and learned theta

        In 2 class cases, label datapoint as class 1 if posterior is greater than .5

        In Multiple class cases, label the class with highest posterior

        """

        if one_v_all:
            print "Decision: one versus all"

            probs = sigmoid(np.dot(X,self.thetas_))
            return np.argmax(probs, axis=1)

        else:
            thershold = 0.5
            hypothesis = sigmoid(np.dot(X,self.theta_))
            # if h > thershold, assign to class 1, to class 0 otherwise
            # add zero to return a 1/0 array instead of boolean array
            return 0 + (hypothesis >= thershold)

    def predict(self,X, one_v_all=False):
        """
        :param
        one_v_all: Boolean, whether in multiple classes cases
        :return:
         m_sample array labels
        """

        #hypothesis = self._hypothesis(X,theta_opt)
        prediction = self.decision_function(X, one_v_all)
        return prediction

def calcCorrect(prediction, target):
    """
    :param prediction: m array
    :param target: m array
    :return: Correct rate
    """
    print "Validating prediction"
    if prediction.shape != target.shape:
        print "Wrong dimension\n"
        print ("pred shape: ", prediction.shape)
        print ("target shape:", target.shape)

    # print prediction
    # print target

    comparison = 0+(prediction ==target)
    print comparison
    return float(sum(comparison)) / len(prediction)


#TODO: assign csvfile of train data
# csvfile =

X,y = load_train(csvfile=csvfile,addConst=True)

# Randomly split data into test and train
splitrate = 0.3
n_sample = X.shape[0]
testRow = random.sample(range(n_sample), int(splitrate * n_sample))
trainRow = [r for r in range(n_sample) if r not in testRow]

X_val = X[testRow, :]
y_val = y[testRow]

X_train = X[trainRow, :]
y_train = y[trainRow]


logit = LogitReg(1 , 1, 400)
logit.fit(X_train,y_train)
logit_p = logit.predict(X_val,True)

print "Accurancy:"
print calcCorrect(logit_p, y_val)


"""
Reference:
Sklearn
http://nbviewer.ipython.org/gist/kevindavenport/c524268ed0713371aa32
http://aimotion.blogspot.com/2011/11/machine-learning-with-python-logistic.html
"""