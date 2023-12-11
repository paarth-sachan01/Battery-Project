import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.io import loadmat
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
np.random.seed(42)

path_9='/Users/paarthsachan/technical/State_of_health_battery/Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab/RW9.mat'
path_10='/Users/paarthsachan/technical/State_of_health_battery/Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab/RW10.mat'
path_11='/Users/paarthsachan/technical/State_of_health_battery/Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab/RW11.mat'
path_12='/Users/paarthsachan/technical/State_of_health_battery/Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab/RW12.mat'
number_of_partial_profiles=1
subarray_length=100
annots = loadmat(path_12)
annots_=annots['data'][0][0]
steps=annots_[0][0]

dict_types=[]
for i in range(len(steps)):
    step=steps[i]
    #print(step[0][0])
    if(step[0][0] not in dict_types):
        dict_types.append(step[0][0])
    #break
  
    
def carve_array_into_subarrays(input_array, number_of_partial_profiles=number_of_partial_profiles
                               ,subarray_length=subarray_length):
    if number_of_partial_profiles <= 0:
        raise ValueError("Number of partial profiles must be greater than 0")

    # Calculate the length of each subarray
    

    # Initialize an empty list to store the subarrays
    skip_between_two_arr=((len(input_array)-
                  subarray_length*number_of_partial_profiles)//number_of_partial_profiles)-1
    subarrays = []

    # Loop to create the subarrays
    for i in range(number_of_partial_profiles):
        start_index = i * subarray_length
        end_index = (i + 1) * subarray_length

        # Append the subarray to the list
        if(i!=0):
            start_index+=i*skip_between_two_arr
            end_index+=i*skip_between_two_arr
        subarrays.append(input_array[start_index:end_index])

    return subarrays


integral_I_dt=[]
Voltage_array_=[]
Current_array_=[]
Time_array_=[]
Temperature_array_=[]
for type_ in dict_types:


    if((type_=='reference discharge')):
        
        train_scores=np.array([])
        test_scores=np.array([])
        all_current=[]
        coverage_scores_arr=np.array([])
        out_bound_scores=np.array([])
        list_=[]
        for i in range(len(steps)):
            if(steps[i][0][0]==type_):
                list_.append(i)
        
        
        for index_current in tqdm(range(len(list_))):
                index_current_=list_[index_current]
                step=steps[index_current_]
                type_=step[0][0]
                time_array=step[3][0]

                non_relative_time=step[2][0]
                voltage_array=step[4][0]#+1e-7
                current_array=step[5][0]#+1e-7
                temperature_array=step[6][0]
                #indexes_distributed=evenly_distributed_indexes(voltage_array)
                Voltage_array_.extend(carve_array_into_subarrays(voltage_array))
                Current_array_.extend(carve_array_into_subarrays(current_array))
                Time_array_.extend(carve_array_into_subarrays(time_array))
                Temperature_array_.extend(carve_array_into_subarrays(temperature_array))
                
                I_dt = np.trapz(current_array, x=time_array)

                for i in range(number_of_partial_profiles):
                    integral_I_dt.append(I_dt/(2.2*3600))

    else:
        pass

# print(len(integral_I_dt))
# number_equal = np.floor(len(integral_I_dt)/3)
# number_middle=len(integral_I_dt) - 2*number_equal
# # Create the arrays with indices
# early_life_indexes = np.arange(0, int(number_equal))
# middle_life_indexes = np.arange(int(number_equal), int(number_equal + number_middle))
# end_life_indexes = np.arange(int(number_equal + number_middle), len(integral_I_dt))

# # Print the lengths of the three arrays
# others_to_train_split=0.2
# test_val_split=0.5
# early_life_train, early_life_temp = train_test_split(early_life_indexes, 
#                                                      test_size=others_to_train_split, random_state=42)
# early_life_validation, early_life_test = train_test_split(early_life_temp, 
#                                                           test_size=test_val_split, random_state=42)

# # Split middle_life indexes
# middle_life_train, middle_life_temp = train_test_split(middle_life_indexes, 
#                                                        test_size=others_to_train_split, random_state=42)
# middle_life_validation, middle_life_test = train_test_split(middle_life_temp, 
#                                                             test_size=test_val_split, random_state=42)

# # Split end_life indexes
# end_life_train, end_life_temp = train_test_split(end_life_indexes, 
#                                                  test_size=others_to_train_split, random_state=42)
# end_life_validation, end_life_test = train_test_split(end_life_temp, 
#                                                       test_size=test_val_split ,random_state=42)

# all_train = np.concatenate((early_life_train, middle_life_train, end_life_train)).tolist()

# # Combine all validation sets
# all_validation = np.concatenate((early_life_validation, 
#                                  middle_life_validation, end_life_validation)).tolist()

# # Combine all test sets
# all_test = np.concatenate((early_life_test, middle_life_test, end_life_test)).tolist()
# print(len(all_train),len(all_validation),len(all_test))
def plotting_seperation():
    all_data = [all_train, all_validation, all_test]

    # Create labels for the histograms
    labels = ['Train', 'Validation', 'Test']


    classifications = []

    # Iterate through numbers from 0 to 79
    for i in range(80):
        if i in all_train:
            for i in range(number_of_partial_profiles):
                classifications.append('Train')
        elif i in all_validation:
            for i in range(number_of_partial_profiles):
                classifications.append('Validation')
        else:
            for i in range(number_of_partial_profiles):
                classifications.append('Test')

    # Print the classifications
    print(classifications)
    colors = {'Train': 'blue', 'Validation': 'green', 'Test': 'red'}

    # Convert classifications to corresponding colors
    segment_colors = [colors[classification] for classification in classifications]

    # Create a bar chart with colored segments
    plt.figure(figsize=(10, 2))
    plt.bar(range(80), np.ones(80), color=segment_colors, edgecolor='black')

    # Set the x-axis and y-axis labels
    plt.xlabel('Index')
    plt.title('Visualizing Classifications with Colors')
    plt.xticks(range(80), range(80))
    plt.yticks([])  # Hide the y-axis

    # Create a legend
    legend_labels = [plt.Rectangle((0, 0), 1, 1, color=colors[classification]) for classification in colors]
    plt.legend(legend_labels, colors.keys())

    # Show the plot
    plt.show()

all_train=[]
all_validation=[]
all_test=[]

for i in range(len(integral_I_dt)):
    if((((i+1)%5)==3)or(((i+1)%5)==4)):
        all_validation.extend([i])
    else:
        all_train.extend([i])


all_train=np.array(all_train)
all_validation=np.array(all_validation)
print(all_train)
print(all_validation)
class CustomDataModule(pl.LightningDataModule):
    def __init__(self, Voltage_array, Current_array, Time_array, Temperature_array, SOH_array, batch_size, val_split=0.2, test_split=0.1, random_state=42):
        super(CustomDataModule, self).__init__()
        self.Voltage_array = Voltage_array
        self.Current_array = Current_array
        self.Time_array = Time_array
        self.Temperature_array = Temperature_array
        self.SOH_array = SOH_array
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.random_state = random_state

        self.X_train = None
        self.y_train = None

        self.X_val =None
        self.y_val = None

    

    def prepare_data(self):
        # Combine input arrays into a single feature tensor
        X = np.stack((self.Voltage_array, self.Current_array, self.Time_array, self.Temperature_array), axis=-1)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(self.SOH_array, dtype=torch.float32)

        # Split the data into train, validation, and test sets
        #print(X.shape,"##############")
        X_train = X[all_train]
        y_train = y[all_train]
        self.X_train=X_train
        self.y_train=y_train

        X_val = X[all_validation]
        y_val = y[all_validation]
        self.X_val=X_val
        self.y_val=y_val

        # Create DataLoader objects for training, validation, and test sets
        self.train_dataset = TensorDataset(X_train, y_train)
        self.val_dataset = TensorDataset(X_val, y_val)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)



## initilize the data module here itself
transfer_learning_data_module = CustomDataModule(Voltage_array_, Current_array_, Time_array_, 
                 Temperature_array_, integral_I_dt,batch_size=240)
# data_module.prepare_data()
# print(len(data_module.SOH_array),len(all_train),len(all_test),len(all_validation))
# print(len(data_module.y_test))
# train_dataloader = data_module.train_dataloader()

# # Calculate the number of steps (batches) in one epoch

# steps_in_one_epoch = len(train_dataloader)

# print(steps_in_one_epoch)