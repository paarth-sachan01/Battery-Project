# Following is the brief description of codebase folders and their roles.
requirement.txt has all the libraries necessary to run the code.
## Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post
Contains the data from the nasa dataset
## General_soh_modelling
Models the soh for any general charging step
data_loading.py is responsible for preprocessing the data before training any model
training.py trains the model and testing.py evaluates it.
Models folder contains different architectures

##Implicit Q learning
Here we have the function which finds the best action(current value) to be chosen when charging to preserve the life of the battery 
data_loading.py is responsible for preprocessing the data before training any model
Models folder contains different architectures
Paper pdf describes the methodology for which I have modified for my purpose 
q_function_model_path.pth is the model with saved trained weights
Load it with architecture QFunction(state_dim=2,action_dim=1,num_categories=number_of_breaks)
This function gives us the expected reward if we have some state and we decide to take a particular action. 
This can be used to find the optimum charging current. We can simply see the expected value for some set of current values and pick for whatever current the expected reward is highest.( In the dataset these sets of values are 0.75A, 1.5A, 2.25A, 3A, 3.75A, 4.5A). 
approximators.py contains different model architectures described in the paper. 
There are other functionalities which have been used in the research paper pdf but are not relevant to our algorithm
## Partial_discharge_modelling
We model the soh from the partial discharge curves 
data_loading.py is responsible for preprocessing the data before training any model
training.py and testing.py are used for training and testing the models.
transfer learning files are used to do finetune to different batteries
## SOH_modeling_discharge
We model the soh from the discharge curves. 
data_loading.py is responsible for preprocessing the data before training any model 
