# assisted_living_pdh_project

The files in this folder were used in the project "AssistedLiving Project – responsible innovations for dignified lives at home for peoplewith mild cognitive impairment or dementia". This is a result from the Ph.D. Thesis "Sensor Event and Activity Prediction using Binary Sensors in Real Homes with Older Adults".
The thesis will be made available after the defense has taken place.

The thesis aim was to identify, apply, and evaluate state-of-the-art prediction methods in real homes of older adults. 
The prediction methods chosen were: Active LeZi (ALZ), SPEED and LSTM neural nets.

The data used in the project is not public and therefore not uploaded here.

We carried on prediction of the next sensor event/activity and time information.

# files in this folder:

-propabilistic methods can be tested with the files SPEED.py and Lezi.py. They require TreePPM.py to run.
-all_methods_accMemory.py applied all the methods (ALZ, SPEED, LSTM nets) to plot graphs of the accuracies for different sizes of memory length.
-data_preparation.py has functions for reading the csv files and for a number of ways to input the data to the algorithms.
-sequential_activity_dataset.py and concurrent_activity_dataset.py have the rules for transforming binary sensor data to activity datasets, as explained in the thesis.
-transfer_learning.py has the script for doing transfer learning across the different apartments to predict the next sensor event.


