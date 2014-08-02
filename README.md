classifier-calibration
======================

Reliability diagrams and calibration with Platt's scaling and isotonic regression.

	adult - a dir containg data, code and results for a random forest and the Adult dataset
	get_diagram_data.py - a helper function for reliability diagrams
	isotonic_regression.py - calibrate a classifier using isotonic regression
	load_data.py - helper for loading data, a Vowpal Wabbit example. You'll need to edit this file.
	load_data_adult.py - loads y and p from random forest trained on the Adult dataset
	log_loss.py - from log_loss import log_loss
	platts_scaling.py - calibrate a classifier using Platt's scaling
	reliability_diagram.py - check your classifier's calibration
	
See [http://fastml.com/calibrating-a-classifier-with-isotonic-regression/](http://fastml.com/calibrating-a-classifier-with-isotonic-regression/) for description.
