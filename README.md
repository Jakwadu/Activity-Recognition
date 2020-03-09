# Activity-Recognition
A collection of scripts for training neural network based classifiers for the challenge of activity recognition from accelerometer data. The data set used can be downloaded from the UCI machine learning datadbase: https://archive.ics.uci.edu/ml/datasets/Activity+Recognition+from+Single+Chest-Mounted+Accelerometer

activityRecognition.py used multiple 1D CNN layers with residual connections whilst activityRecognition_LSTM.py uses a single LSTM layer. Both models use two dense layers for classification with 800 and 400 nodes each.

# References
Casale, P. Pujol, O. and Radeva, P.
'Personalization and user verification in wearable systems using biometric walking patterns'
Personal and Ubiquitous Computing, 16(5), 563-580, 2012
