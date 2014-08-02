#!/usr/bin/env python

"plot a reliability diagram showing how good a classifier's calibration is"

import numpy as np
import matplotlib.pyplot as plt

from get_diagram_data import get_diagram_data

#from load_data import y, p
from load_data_adult import y, p

print "computing..."

n_bins = 20

mean_predicted_values, true_fractions = get_diagram_data( y, p, n_bins )	
plt.plot( mean_predicted_values, true_fractions )

# perfect calibration line
plt.plot( np.linspace( 0, 1 ), np.linspace( 0, 1 ), 'gray' )

plt.show()
