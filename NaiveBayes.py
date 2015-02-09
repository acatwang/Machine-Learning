"""
Naive Bayes Classifier

with customized features extracter

practice with SMSSpamCollection dataset
http://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
"""
#Author: Alison Wang(yiyin)

import numpy as np
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import names
import codecs,re

my_stopwords = { u"(",u")",u"'", u"&",u".",u",",u".." #meaneingless punctuation
                u've',u'll',u"don",u'm', u're', #fix tokenizeer
                u'u',u'ur',
                u'get','got',
                u'me', u'my', u'myself', u'we', u'our', u'ours', u'ourselves', u'you', u'your', u'yours', u'yourself', u'yourselves', u'he', u'him', u'his', u'himself', u'she', u'her', u'hers', u'herself', u'it', u'its', u'itself', u'they', u'them', u'their', u'theirs', u'themselves',
                 u'what',u'which', u'who', u'whom', u'this', u'that', u'these', u'those', u'am', u'is', u'are', u'was', u'were', u'be', u'been', u'being', u'have', u'has', u'had', u'having', u'do', u'does', u'did', u'doing', u'a', u'an', u'the', u'and', u'but', u'if', u'or', u'because', u'as', u'until', u'while', u'of', u'at', u'by', u'for', u'with', u'about', u'against', u'between', u'into', u'through', u'during', u'before', u'after', u'above', u'below', u'to', u'from', u'up', u'down', u'in', u'out', u'on', u'off', u'over', u'under', u'again', u'further', u'then', u'once', u'here', u'there', u'when', u'where', u'why', u'how', u'all', u'any', u'both', u'each', u'few', u'more', u'most', u'other', u'some', u'such', u'no', u'nor', u'not', u'only', u'own', u'same', u'so', u'than', u'too', u'very', u's', u't', u'can', u'will', u'just', u'don', u'should', u'now'}

def load_data(path,clean=True):
    """
    Read text file
    :return:
    bag_or_words
    label
    """
    with codecs.open(path, 'rU',encoding='utf-8') as con:
        y = []
        X = []
        X_noStop = []
        for line in con:
            label, text = line.split('\t')
            label = 1 if label=="spam" else 0
            y.append(label)

            words = wordpunct_tokenize(text)
            X.append(words)
            X_noStop.append([w for w in words if w.lower() not in my_stopwords and re.search('[a-zA-Z]', w)])

        X = np.asarray(X)
        X_noStop = np.asarray(X_noStop)
        y = np.asarray(y)
        print "Load X: ", X.shape
        print X[0:5]
        print "Load X no stopwords:", X_noStop.shape
        print X_noStop[0:5]
        print "Load y: ", y.shape
        print y[0:5]

        data = X_noStop if clean else X
        return data, y

mail_train, label_train = load_data("hw2_datasets/SMSSpamCollection_train.txt")
mail_test, label_test = load_data("hw2_datasets/SMSSpamCollection_test.txt")

# Count words

word_cnt ={}

def countWords(isSpam,X,y):

    cls = isSpam + 0
    cntDict = {}
    for line in X[y==cls]:
        for word in line:
            word = word.lower()
            if word in cntDict:
                cntDict[word] +=1
            else:
                cntDict[word] =1

            if word in word_cnt:
                word_cnt[word] +=1
            else:
                word_cnt[word] =1
    return cntDict

spam_word_cnt = countWords(True,mail_train,label_train)
ham_word_cnt = countWords(False,mail_train,label_train)

print sorted(spam_word_cnt.items(), key=lambda x:x[1], reverse=True)[0:30]
print sorted(ham_word_cnt.items(), key=lambda x:x[1],reverse=True)[0:30]


# Construct features
def hasName(words):
    for w in words:
        if w in names.words():
            return True
    return False

def countAllCapital(words):
    cnt = 0
    for w in words:
        if w.isupper():
            cnt+=1
    return cnt

def conutPercentage(inputWords, targetWord):
    x = 0.0+ len([w for w in inputWords if w.lower()== targetWord])
    freq = x/len(targetWord)
    return 100 * freq

def feature_extractor(words):
    """
    :param sentence:
    :return: n_features array

    # True false
    ============
    # hasName

    ===========
    # Numeric
    ============
    # Length of sentence
    # number of capital char

    # Number of times the following word shows in sentence
    ==============
    # "Call"
    # "Free"
    # "stop"
    # "prize"
    # "text"
    # "claim"
    # "reply"
    # "win", "won"
    # "cash"
    # "new"
    # "service"
    # "please"
    # "urgent"
    # "www"
    # "uk"
    # "msg"
    # go
    # "love"
    # "sorry"
    # home"
    # "com"
    # "!"
    # today

    """

    record = [0]*26

    if hasName(words):
        record[0] = 1
    record[1] = len(words)
    record[2] = countAllCapital(words)
    record[3] = conutPercentage(words,'call')
    record[4] = conutPercentage(words,'free')
    record[5] = conutPercentage(words,'stop')
    record[6] = conutPercentage(words,'prize')
    record[7] = conutPercentage(words,'txt')
    record[8] = conutPercentage(words,'claim')
    record[9] = conutPercentage(words,'reply')
    record[10] = conutPercentage(words,'win') + conutPercentage(words,"won")
    record[11] = conutPercentage(words,'cash')
    record[12] = conutPercentage(words,'new')
    record[13] = conutPercentage(words,'service')
    record[14] = conutPercentage(words,'please')
    record[15] = conutPercentage(words,'urgent')

    record[16] = conutPercentage(words,'www')
    record[17] = conutPercentage(words,'uk')
    record[18] = conutPercentage(words,'msg')
    record[19] = conutPercentage(words,'go')
    record[20] = conutPercentage(words,'love')
    record[21] = conutPercentage(words,'sorry')
    record[22] = conutPercentage(words,'home')
    record[23] = conutPercentage(words,'com')
    record[24] = conutPercentage(words,'!')
    record[25] = conutPercentage(words,'today')

    return record


def construct_feature_table(mail):
    X = []
    for words in mail:
        X.append(feature_extractor(words))

    X = np.array(X)
    print "Converted raw data, shape: ", X.shape

    return X

# Fit and train classifier

class NaiveBayesClassfier(object):
    def __init__(self):
        pass

    def fit(self,X_train, y_train):
        num_span = np.sum(y_train)
        num_Nspan =  y_train.size-num_span

        print num_span
        print num_Nspan

        # Print Spam prior
        p_spam = np.mean(y_train)
        print "Spam prior: ", p_spam
        spams= X_train[np.where(y_train==1)]
        Nspams = X_train[np.where(y_train==0)]

        print spams.shape
        print np.sum(spams)
        print float(np.sum(spams))

        prob_vec_spam = (np.sum(spams, axis=0)+0.0) / (num_span+0.0)
        prob_vec_Nspam = (np.sum(Nspams,axis=0)+0.0) / (num_Nspan+0.0)


        self.p_spam = p_spam
        self.p_Nspam = 1-p_spam
        self.prob_vec_spam = prob_vec_spam
        self.prob_vec_Nspam = prob_vec_Nspam

        return self

    def predict(self,X):
        """
        Calculate log probability
        Return label array
        """
        log_prob_spam = np.log(self.p_spam) + X.dot(np.log(self.prob_vec_spam))
        log_prob_Nspam = np.log(self.p_Nspam) + X.dot(np.log(self.prob_vec_Nspam))

        return 0+(log_prob_spam > log_prob_Nspam)


X_train = construct_feature_table(mail_train)
X_test = construct_feature_table(mail_test)
def quantizeData(X_train,X_test):

    # Quantize data
    X_all = np.concatenate((X_train,X_test))
    print X_all.shape

    # Calculate median values of all features using test and train data
    median_vec = np.median(X_all,axis=0)
    print median_vec.shape

    # Quantize
    # Map values below median to 0, 1 otherwise
    X_train = 0 + (X_train > median_vec)
    X_test = 1 + (X_test > median_vec)

    return X_train, X_test

X_train, X_test = quantizeData(X_train,X_test)

nbClf = NaiveBayesClassfier()
nbClf.fit(X_train, label_train)
predict = nbClf.predict(X_test)

print "Accurancy: ", np.mean(0+(predict == label_test))



"""
Reference:
http://nbviewer.ipython.org/github/carljv/Will_it_Python/blob/master/MLFH/CH3/ch3_nltk.ipynb
http://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
"""