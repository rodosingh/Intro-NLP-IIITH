The Code file to be executed for Launching the Baseline model is :Hypernym_discovery_baseline_implementation.py

The requirements to successfully exeute the file are :
1) Python version : Python 3.9.5
2) GCC version : 7.5.0
3) Numpy version: 1.19.2
4) Tensorflow version :2.6.2

The Tasks are defined as follows :
Task 1A:Training the Baseline models on English data 
Task 2A:Training the Baseline models on Medical data
TAsk 2B:Training the Baseline models on Music data

To execute the `Hypernym_discovery_baseline_implementation.py` successfully following steps needs to be followed :

Step 1: Call the hypernym_discovery_baseline(task, model) from within the Hypernym_discovery_baseline_implementation.py file
Step 2: The 2 arguments accept the following value :
1. task 
	1. task ="1A" ( to train the baseline on English data)
	2. task ="2A" ( to train the baseline on Medical data)
	3. task ="2B" ( to train the baseline on Music data)

2. model 
	1. model="gru" ( to train the gru model on the selected corpus using task argument)
	2. model ="lstm" ( to train the  lstm model on the selected corpus using task argument)


Note: Kindly caliberate the Directory and file paths prior to execution of the Script.
