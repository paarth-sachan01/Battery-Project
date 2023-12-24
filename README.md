# Following is the brief description of codebase folders and their roles.
requirement.txt has all the libraries necessary to run the code.
## Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post
Contains the data from the nasa dataset
## General_soh_modelling
Models the soh for any general charging step
data_loading.py is responsible for preprocessing the data before training any model
training.py trains the model and testing.py evaluates it.
Models folder contains different architectures

## Implicit Q learning
Here we have the function which finds the best action(current value) to be chosen when charging to preserve the life of the battery 
## Partial_discharge_modelling
We model the soh from the partial discharge curves 
data_loading.py is responsible for preprocessing the data before training any model
training.py and testing.py are used for training and testing the models.
transfer learning files are used to do finetune to different batteries
## SOH_modeling_discharge
We model the soh from the discharge curves. 
data_loading.py is responsible for preprocessing the data before training any model 
