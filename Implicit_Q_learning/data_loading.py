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
import torch
from torch.utils.data import Dataset, DataLoader

np.random.seed(42)

path_9='/Users/paarthsachan/technical/State_of_health_battery/Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab/RW9.mat'
path_10='/Users/paarthsachan/technical/State_of_health_battery/Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab/RW10.mat'
path_11='/Users/paarthsachan/technical/State_of_health_battery/Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab/RW11.mat'
path_12='/Users/paarthsachan/technical/State_of_health_battery/Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab/RW12.mat'
number_of_partial_profiles=1
subarray_length=150
annots = loadmat(path_9)
annots_=annots['data'][0][0]
steps=annots_[0][0]

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


last_reference_discharge=0
reference_discharge_indexes=[]
list_=[]
for i in range(len(steps)):
    step=steps[i]
    #print(step[0][0])
    list_.append(i)
    type_=step[0][0]
    if(type_=='reference discharge'):
        last_reference_discharge=i
        reference_discharge_indexes.append(i)

Voltage_array_=[]
Current_array_=[]
Time_array_=[]
Temperature_array_=[]


time_stitched= []
current_stitched= []
voltage_stitched= []
temperature_stitched= []
impedence_stitched= []
depth_discharge_stitched= []

depth_discharge_running=0
dod_outside_range=0
dod_outside_range_index=[]
#upper_bound_soh=[1]
lower_bound_soh=[]
num_datapoints=[]
charging_index=[]

for i in range(last_reference_discharge+1):
    step=steps[i]
    #print(step[0][0])
    type_=step[0][0]
    time_array=step[3][0]

    non_relative_time=step[2][0]
    voltage_array=step[4][0]
    current_array=step[5][0]
    temperature_array=step[6][0]
    impedence_array= (voltage_array+1e-7)/(current_array+1e-7)
    Voltage_array_.extend(carve_array_into_subarrays(voltage_array))
    Current_array_.extend(carve_array_into_subarrays(current_array))
    Time_array_.extend(carve_array_into_subarrays(time_array))
    Temperature_array_.extend(carve_array_into_subarrays(temperature_array))
    
    if((type_=='reference charge')or(type_=='charge (random walk)')):
        charging_index.append(i)

    depth_discharge_stitched.append(depth_discharge_running)
    for i in range(1,len(time_array)):
        if(depth_discharge_running>22):
            dod_outside_range+=1
            dod_outside_range_index.append(i)

        depth_discharge_running+=(np.trapz(current_array[i-1:i+1], 
                                    x=time_array[i-1:i+1])/3600).item()
        
        depth_discharge_stitched.append(depth_discharge_running)
        
    #stitching procedure
    voltage_stitched.extend(voltage_array)
    current_stitched.extend(current_array)
    time_stitched.extend(non_relative_time)
    temperature_stitched.extend(temperature_array)
    impedence_stitched.extend(impedence_array)


    if(type_=='reference discharge'):
        #x1= upper_bound_soh[-1]
        soh_estimate=np.trapz(current_array, x=time_array)/(2.2*3600)
        lower_bound_soh.append(soh_estimate)
    else:
        lower_bound_soh.append(-1)

    num_datapoints.append(len(time_array))

window_size=5
relevant_indicies=[]
for i in range(len(reference_discharge_indexes)):
    start_idx = reference_discharge_indexes[i]-window_size
    end_idx = reference_discharge_indexes[i] + window_size
    array_of_relevant = list_[start_idx:end_idx]
    relevant_indicies.extend(array_of_relevant)

temp_relevant_indicies= []
type_of_relevant= []
for i in range(len(relevant_indicies)):
    global_ind=relevant_indicies[i]
    step=steps[global_ind]
    type_=step[0][0]
    time_array=step[3][0]
    if(len(time_array)>=subarray_length):
        temp_relevant_indicies.append(global_ind)
        type_of_relevant.append(type_)

relevant_indicies=temp_relevant_indicies
relevant_indicies = list(set(relevant_indicies) & set(charging_index))


next_lower_bound=None
count=0
print("###",len(lower_bound_soh))
for i in range(len(lower_bound_soh)):
    if(i<reference_discharge_indexes[0]):
        lower_bound_soh[i]=lower_bound_soh[reference_discharge_indexes[0]]
    else:
        if(lower_bound_soh[i] == -1):
            lower_bound_soh[i]=lower_bound_soh[reference_discharge_indexes[count]]

        else:
            count+=1


soh_estimate_stitched = [elem for elem, count_ in zip(lower_bound_soh, num_datapoints) for _ in range(count_)]

number_of_breaks=20
chunk_size=100//number_of_breaks
full_battery_value=40
discretised_values =[]
for i in range(len(soh_estimate_stitched)):
    fraction_remaining = (((soh_estimate_stitched[i]*100)//chunk_size)+1)/number_of_breaks
    value_step= int(fraction_remaining*full_battery_value)
    discretised_values.append(value_step)
    ## example 0.9 soh and 5 chunch size-> fract remaining =((90//20)+1)/5==1
    ## it is important to divide 2 times for proper descretisation 
    ### int is done as fraction*initial value is a float and can be a integer sometimes
counts = {}

# Iterate through the array and update the counts
for value in discretised_values:
    if value in counts:
        counts[value] += 1
    else:
        counts[value] = 1

# Print the counts
for value, count in counts.items():
    print(f"{value}: {count}")




# for i in range(len(time_stitched)):
#     discharged = np.trapz(current_stitched[:i+1], 
#                                 x=time_stitched[:i+1]).item()
#     depth_discharge_stitched.append(discharged)


######
# depth_discharge_running=0
# depth_discharge_stitched.append(depth_discharge_running)
# ## divided by 3600 for the depth of discharge in Amp Hour
# for i in range(1,len(time_stitched)):
#     depth_discharge_running+=(np.trapz(current_stitched[i-1:i+1], 
#                                 x=time_stitched[i-1:i+1])/3600).item()
#     depth_discharge_stitched.append(depth_discharge_running)

                

print(len(time_stitched),len(current_stitched),
      len(voltage_stitched),len(temperature_stitched),
      len(impedence_stitched),len(soh_estimate_stitched))



# print(min(depth_discharge_stitched),max(depth_discharge_stitched))
# print(relevant_indicies,"RELEVANT INDICIES")
# print(jd)
class CustomDataset(Dataset):
    def __init__(self, time_, current_, voltage_, 
                 temperature_):
        self.time_stitched = time_
        self.current_stitched =current_
        self.voltage_stitched = voltage_
        self.temperature_stitched = temperature_
        self.total_samples = len(self.current_stitched)

    def __len__(self):
        return self.total_samples-2

    def __getitem__(self, idx_starting):
        start_idx = idx_starting
        end_idx = start_idx + 1
        end_idx2= end_idx+1

        current_chunk = self.current_stitched[start_idx:end_idx]
        voltage_chunk = self.voltage_stitched[start_idx:end_idx]
        temperature_chunk = self.temperature_stitched[start_idx:end_idx]


        voltage_chunk2 = self.voltage_stitched[end_idx:end_idx2]
        temperature_chunk2 = self.temperature_stitched[end_idx:end_idx2]
        

        state_1 =  torch.stack((torch.tensor(voltage_chunk), 
                              torch.tensor(temperature_chunk)),dim=2)
        state_2 =  torch.stack((torch.tensor(voltage_chunk2), 
                             torch.tensor(temperature_chunk2)),dim=2)
        state_1 = torch.squeeze(state_1)
        state_2 = torch.squeeze(state_2)
        
        action =  torch.tensor(current_chunk)

        state_action = torch.stack((torch.tensor(voltage_chunk), 
                        torch.tensor(current_chunk), torch.tensor(temperature_chunk)),dim=2)
        state_action= torch.squeeze(state_action)
        return state_1.to(torch.float32), action.to(torch.float32) ,state_2.to(torch.float32), state_action.to(torch.float32)


# Now, you can create a DataLoader using your custom dataset

new_Time_array = [Time_array_[i] for i in relevant_indicies]
new_Current_array = [Current_array_[i] for i in relevant_indicies]
new_Voltage_array = [Voltage_array_[i] for i in relevant_indicies]
new_Temperature_array = [Temperature_array_[i] for i in relevant_indicies]

# print(len(new_Time_array),"$$$$$$$$$$$$$")
# print(jnsakd)
dataset = CustomDataset(new_Time_array, new_Current_array, new_Voltage_array, new_Temperature_array)

