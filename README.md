# Split Neural Network Federated Learning

The goal of this project is to make a Federated Algorithm in order to obtain a classification on two labels : Location and faultLabel.
I am working with four different datasets.
They are composed as follows : 
- Divided into four groups :
  - df1 : with shape (968,51)
  - df2 : with shape (968,51)
  - df3 : with shape (968,51)
  - df4 : with shape (968,51)
  
We obtained this dataset by injecting fault with different resistances on four different zones in the IEEE-13 distribution network with renewable energy. 
The columns are : 
 - voltage : signal measured
 - measloc : zone from which the voltage has been measured (between 1 and 4 (IEEE-13))
 - locLabel : zone in which the fault has been injected (1-4)
 - resistance : shows the different number of resistances
 - faultLabel : represents the type of the faults (11 types : ABC, ABCG, AB, AC, BC, ABG, ACG, BCG, AG, BG, CG)

# Split Neural Network with combination of fault and location

In order to see what would have changed using another kind of prediction, I decided to use the same Split Neural Network, but instead of computing two different accuracies, i chose to add a combination of fault and location.

Here below you can see the results

![immagine](https://user-images.githubusercontent.com/96230284/149166690-749fa93e-bff6-4f0c-ae5f-9af7aa40a9fc.png)

# Single Locations algorithm 

At this point, i need to have a baseline, in order to judge the goodness of the results.
So for this task, i chose to use an algorithm for single locations, where i calculate the accuracy of the prediction for each location.
To make it more clear, i'll show here below an example of the output of the code : 

<img width="186" alt="eslocations" src="https://user-images.githubusercontent.com/96230284/149161520-064d5b2b-2014-49e6-95a6-a8e96ab558dd.PNG">

# Result obtained

Here below there is a comparation between al the results obtained tuning the hyperparameters

![immagine](https://user-images.githubusercontent.com/96230284/146537458-0435e1cd-4161-4241-bae5-f50ab0d3dea8.png)
