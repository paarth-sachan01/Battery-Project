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

annots = loadmat(path_9)
annots_=annots['data'][0][0]
steps=annots_[0][0]

dict_types=[]
for i in range(len(steps)):
    step=steps[i]
    #print(step[0][0])
    if(step[0][0] not in dict_types):
        dict_types.append(step[0][0])
    #break
  
    
def evenly_distributed_indexes(arr, num_indexes=200,convolve=False,window_size=3):
    # print(arr)
    if(convolve==True):
        # arr=np.convolve(arr, np.ones(window_size)/window_size, 
        #                                 mode='valid').tolist()
        arr_ = list(map(lambda i: np.mean(arr[max(i - window_size + 1, 0):i + 1]), range(len(arr))))
        # print(indexes)       
        # #arr= arr[indexes]
        # print(jd)
    else:
        arr_ = arr

    if len(arr) <= num_indexes:
        return arr_

    step = len(arr_) // num_indexes  # Calculate the step size for even distribution
    arr_fin = [arr_[i * step] for i in range(num_indexes)]  # Generate evenly distributed indexes
    
    #arr_=arr[indexes]
    return arr_fin
    #return indexes

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
                Voltage_array_.append(evenly_distributed_indexes(voltage_array))
                Current_array_.append(evenly_distributed_indexes(current_array))
                Time_array_.append(evenly_distributed_indexes(time_array))
                Temperature_array_.append(evenly_distributed_indexes(temperature_array))
                
                I_dt = np.trapz(current_array, x=time_array)
                integral_I_dt.append(I_dt/(2.2*3600))

    else:
        pass
plt.plot(integral_I_dt)
plt.show()
print(jd)
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
            classifications.append('Train')
        elif i in all_validation:
            classifications.append('Validation')
        else:
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
for i in range(80):
    if(((i+1)%5)==2):
        all_validation.append(i)
    elif (((i+1)%5)==4):
        all_test.append(i)
    else:
        all_train.append(i)


all_train=np.array(all_train)
all_validation=np.array(all_validation)
all_test=np.array(all_test)

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

    def prepare_data(self):
        # Combine input arrays into a single feature tensor
        X = np.stack((self.Voltage_array, self.Current_array, self.Time_array, self.Temperature_array), axis=-1)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(self.SOH_array, dtype=torch.float32)

        # Split the data into train, validation, and test sets
        X_train = X[all_train]
        y_train = y[all_train]

        X_val = X[all_validation]
        y_val = y[all_validation]

        X_test = X[all_test]
        y_test = y[all_test]
        # Create DataLoader objects for training, validation, and test sets
        self.train_dataset = TensorDataset(X_train, y_train)
        self.val_dataset = TensorDataset(X_val, y_val)
        self.test_dataset = TensorDataset(X_test, y_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


## initilize the data module here itself
data_module = CustomDataModule(Voltage_array_, Current_array_, Time_array_, 
                 Temperature_array_, integral_I_dt,batch_size=80)

# train_dataloader = data_module.train_dataloader()

# # Calculate the number of steps (batches) in one epoch

# steps_in_one_epoch = len(train_dataloader)

# print(steps_in_one_epoch)