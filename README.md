# Ma412_Project
This depository contains the programs used in the project

ABDELMALEK Enzo
ROBINEAU ELiott
SIALELLI Janelle

This file explains the goal of the project and gives the directions to use the the programs

### Goal of the projet ###

The goal of this project is to realize an automated mutli-label classification to classify scientific papers and organize them in an architecture where we can find articles related to each others.

### Explanation of the programs ###

The program is splited into three distinct files, namely Ma412_lib.py, Ma412_main.py and Ma412_class.py containing respectively the functions used for this project, the examples shown in the report and the class used. 

Please before trying any model in the main programm compile the loading data section and the preprocessing one. Those sections are required for the following of the program and you won't be able to use it if you don't compile them.

Watch out when you run the second method, some error can occur if you try to run the first one after. The problem can come from the value of N_train that does not change. If you are affected by this error, please restart the kernel and reload the data.

To run the LogisticRegression model, we provide you the model we computed. All you have to do is to load it. If you want to test our algorithm you have to uncomment the line 168 and if you want to save it the lines 170 to 172.
