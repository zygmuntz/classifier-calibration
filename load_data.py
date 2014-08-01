#!/usr/bin/env python

import numpy as np
from scipy.special import expit as sigmoid

# this example shows how to load y from a test file in VW format
# and predictions too - VW doesn't output probabilities hence the sigmoid
# we want y like np.array([ 0, 1, 0, 1 ])
# and p like np.array([ 0.3, 0.98, 0.2, 0.75435832345 ])

test_file = 'test_v.vw'
predictions_file = 'p.txt'

print "loading data..."

y = np.loadtxt( test_file, usecols = [ 0 ] )

# y need to be 0/1
y[y == -1] = 0

p = sigmoid( np.loadtxt( predictions_file, usecols = [ 0 ] ))

assert( y.shape[0] == p.shape[0] )
