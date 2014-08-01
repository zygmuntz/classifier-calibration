classifier-calibration
======================

Reliability diagrams and calibration with isotonic regression.

	adult - a dir containg data, code and results for a random forest and the Adult dataset
	isotonic_regression.py - calibrate a classifier
	load_data.py - helper for loading data, a Vowpal Wabbit example. You'll need to edit this file.
	load_data_adult.py - loads y and p from random forest trained on the Adult dataset
	log_loss.py - from log_loss import log_loss
	reliability_diagram.py - check your classifier's calibration
	
See [http://fastml.com/calibrating-a-classifier-with-isotonic-regression/](http://fastml.com/calibrating-a-classifier-with-isotonic-regression/) for description.
