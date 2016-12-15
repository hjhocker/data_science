#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 12:22:44 2016

@author: harrisonhocker
"""
from scipy import linalg

import os
import sys
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
import math

plt.style.use('ggplot')

#Generate sample vectors
v = np.array([1, 4, 6, 2, 90])
w = np.array([84, 34, 88, 1, 7])

#Calculate the normal (distance) of a vector
print linalg.norm(w)
print math.sqrt(84*84 + 34*34 + 88*88 + 1*1 + 7*7)

#Calculate the distance between two vectors
print linalg.norm(v-w)
print math.sqrt((1-84)**2 + (4-34)**2 + (6-88)**2 + (2-1)**2 + (90-7)**2)

#Calculate dot product and outer product of two vectors
print v.dot(w)
print np.outer(v, w)

print "Covariance matrix"
n, p = 10, 4
v = np.random.random((p, n))
print v
print "The numpy covariance matrix"
print np.cov(v)

print "Manually calculate the covariance matrix"
# From the definition, the covariance matrix
# is just the dot product of the normalized
# matrix where every variable has zero mean
# divided by the number of degrees of freedom
w = v - v.mean(1)[:, np.newaxis] #normalize the vector
print w.dot(w.T)/(n - 1)