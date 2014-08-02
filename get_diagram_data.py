import numpy as np

def get_diagram_data( y, p, n_bins ):

	n_bins = float( n_bins )	# a float to take care of division

	# we'll append because some bins might be empty
	mean_predicted_values = np.empty(( 0, ))
	true_fractions = np.zeros(( 0, ))

	for b in range( 1, int( n_bins ) + 1 ):
		i = np.logical_and( p <= b / n_bins, p > ( b - 1 ) / n_bins )	# indexes for p in the current bin
		
		# skip bin if empty
		if np.sum( i ) == 0:
			continue

		mean_predicted_value = np.mean( p[i] )
		# print "***", np.sum( y[i] ), np.sum( i )
		true_fraction = np.sum( y[i] ) / np.sum( i )					# y are 0/1; i are logical and evaluate to 0/1

		print mean_predicted_value, true_fraction

		mean_predicted_values = np.hstack(( mean_predicted_values, mean_predicted_value ))
		true_fractions = np.hstack(( true_fractions, true_fraction ))
		
	return ( mean_predicted_values, true_fractions )
	