# VerticalFL

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

![immagine](https://user-images.githubusercontent.com/96230284/146537458-0435e1cd-4161-4241-bae5-f50ab0d3dea8.png)
