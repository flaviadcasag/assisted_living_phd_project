# assisted_living_phd_project

The files in this folder were used in the project "AssistedLiving Project – responsible innovations for dignified lives at home for peoplewith mild cognitive impairment or dementia". This is a result from the Ph.D. Thesis "Sensor Event and Activity Prediction using Binary Sensors in Real Homes with Older Adults".<br />
The thesis will be made available after the defense has taken place.<br />

The thesis aim was to identify, apply, and evaluate state-of-the-art prediction methods in real homes of older adults. <br />
The prediction methods chosen were: Active LeZi (ALZ), SPEED and LSTM neural nets.

The data used in the project is not public and therefore not uploaded here.<br />

We carried out prediction of the next sensor event/activity and time information.<br />

# files in this folder:

- propabilistic methods can be tested with the files SPEED.py and Lezi.py. They require TreePPM.py to run. <br />
- all_methods_accMemory.py applied all the methods (ALZ, SPEED, LSTM nets) to plot graphs of the accuracies for different sizes of memory length. <br />
- data_preparation.py has functions for reading the csv files and for a number of ways to input the data to the algorithms. <br />
- sequential_activity_dataset.py and concurrent_activity_dataset.py have the rules for transforming binary sensor data to activity datasets, as explained in the thesis. <br />
- transfer_learning.py has the script for doing transfer learning across the different apartments to predict the next sensor event.


