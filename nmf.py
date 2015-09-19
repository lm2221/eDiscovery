import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class NMF(object):
    def fit(self, X, d=25, max_iter=200, printstuff=False):
        N,M = X.shape
        self.W = np.random.random((N,d))
        self.H = np.random.random((d,M))
        objective = np.zeros(max_iter)

        for t in xrange(max_iter):
            objective[t] = self.objective(X)
            self.update(X)
            if printstuff:
                print "t =",t, "objective =",objective[t]
        return objective



class NMFSquare(NMF):
    def objective(self, X):
        diff = X - np.dot(self.W,self.H)
        return (diff**2).sum() # element-by-element square

    def update(self, X):
        A = np.dot(self.W.T,X)
        B = np.dot(np.dot(self.W.T, self.W), self.H)
        self.H *= A/B

        C = np.dot(X,self.H.T)
        D = np.dot(np.dot(self.W, self.H), self.H.T)
        self.W *= C/D

class NMFDivergence(NMF):
    def __init__(self, epsilon=10**-16):
        self.epsilon = epsilon

    def objective(self, X):
        dot = np.dot(self.W,self.H) + self.epsilon
        diff = X*np.log(dot) - dot
        return diff.sum()

    def update(self, X):
        # update H first
        N,M = X.shape
        W = self.W
        H = self.H
        D = self.W.shape[1]
        dot = np.dot(W,H) + self.epsilon
        top = np.zeros(H.shape)

        for i in xrange(N):
            top += np.dot(W[i].reshape((D,1)), (X[i]/dot[i]).reshape((1,M)))
      
        bottom = W.sum(axis=0) # need to turn this into a matrix - repeat it M times
        bottom += self.epsilon
        bottom = np.dot(bottom.reshape((D,1)), np.ones((1,M)))
        H = H*top/bottom
        self.H = H

        if np.any(np.isnan(top)):
            raise Exception("NaN in top H")

        # now update W
        dot = np.dot(W,H) + self.epsilon
        top = np.zeros(W.shape)

    
        for j in xrange(M):
            top += np.dot( H[:,j].reshape((D,1)), (X[:,j]/dot[:,j]).reshape((1,N)) ).T

        bottom = H.sum(axis=1) # turn this into a matrix - repeat it D times
        bottom += self.epsilon
        bottom = np.dot(np.ones((N,1)), bottom.reshape(1,D))
        W = W*top/bottom
        self.W = W

        if np.any(np.isnan(top)):
            raise Exception("NaN in top W")

        if np.any(np.isnan(W)) or np.any(np.isnan(H)):
            raise Exception("NaN in W or H")


if '1' in sys.argv:
    faces = pd.read_csv('hw5text/faces.csv', header=None).as_matrix()
    nmf = NMFSquare()
    objective = nmf.fit(faces)
    plt.plot(objective)
    plt.show()

    # show 3 images with best W column
    images_shown = []
    N,M = faces.shape
    while True:
        j = np.random.randint(M)
        if j not in images_shown:
            d = nmf.H[:,j].argmax()
            other_face = nmf.W[:,d]
            face = faces[:,j]
            plt.subplot(121)
            plt.imshow(face.reshape((32,32)).T, cmap = cm.Greys_r)
            plt.title("Original image")
            plt.subplot(122)
            plt.imshow(other_face.reshape((32,32)).T, cmap = cm.Greys_r)
            plt.title("Image from W")
            plt.show()

            images_shown.append(j)
        if len(images_shown) >= 3:
            break

if '2' in sys.argv:
    # we are given that X is 3012 x 8447
    N = 3012
    M = 8447
    X = np.zeros((N,M))
    for j,line in enumerate(open('hw5text/nyt_data.txt')):
        # j indexes document, i indexes word
        row = line.split(',')
        for wordcount in row:
            word, count = [int(x) for x in wordcount.split(':')]
            X[word-1,j] = count

    print "Finished loading data"
    nmf = NMFDivergence()
    objective = nmf.fit(X, printstuff=True)
    plt.title('Objective')
    plt.plot(objective)
    plt.show()

    # pick 5 columns of W, choose the top 10 words
    all_words = []
    for line in open('hw5text/nytvocab.dat'):
        all_words.append(line.rstrip())

    for k in np.random.choice(range(25), 5):
        max_indices = (-nmf.W[:,k]).argsort()
        print "Top 10 words in this category:"
        for i in xrange(10):
            print "\t%s" % all_words[max_indices[i]]

