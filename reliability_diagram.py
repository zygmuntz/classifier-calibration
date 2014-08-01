#!/usr/bin/env python

"plot a reliability diagram showing how good a classifier's calibration is"

import numpy as np
import matplotlib.pyplot as plt

#from load_data import y, p
from load_data_adult import y, p

print "computing..."

n_bins = 20.0	# a float to take care of division

mean_predicted_values = np.zeros(( n_bins ))
true_fractions = np.zeros(( n_bins ))

for b in range( 1, int( n_bins ) + 1 ):
	i = np.logical_and( p <= b / n_bins, p > ( b - 1 ) / n_bins )	# indexes for p in the current bin

	mean_predicted_value = np.mean( p[i] )
	true_fraction = np.sum( y[i] ) / np.sum( i )					# y are 0/1; i are boolean and evaluate to 0/1

	print mean_predicted_value, true_fraction

	mean_predicted_values[b - 1] = mean_predicted_value
	true_fractions[b - 1] = true_fraction
	
plt.plot( mean_predicted_values, true_fractions )

# perfect calibration line
plt.plot( np.linspace( 0, 1 ), np.linspace( 0, 1 ), 'gray' )

plt.show()
