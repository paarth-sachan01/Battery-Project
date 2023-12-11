import torch
import numpy as np
import torch.optim as optim
from approximators import QFunction, ValueFunction ,PolicyNetwork
from utils import ExpectileLoss,Exponential_loss
from data_loading import number_of_breaks,full_battery_value, dataset
from torch.distributions.normal import Normal
import torch
#from Models.lstm_model import LSTMModel
#from Models.tranformer import TransformerModel
from Models.only_lstm import Only_LSTMModel

from torch.utils.data import Dataset, DataLoader
# from tensorboard import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm  #

writer = SummaryWriter('logs')

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device =torch.device("cpu")


batch_size=10

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
num_epochs=100000


path ='/Users/paarthsachan/technical/State_of_health_battery/General_soh_modelling/checkpoints/best_model-v31.ckpt'
# path ='/Users/paarthsachan/technical/State_of_health_battery/General_soh_modelling/best_model/best_model_checkpoint.pth'
best_model_trained = Only_LSTMModel.load_from_checkpoint(path,
                        input_size=3,  hidden_size=8,
                       class_to_range=None, num_layers=1, 
                        output_size=2).to(device)
best_model_trained.eval()


Q_function = QFunction(state_dim=2,action_dim=1,num_categories=number_of_breaks).to(device)
V_function = ValueFunction(state_dim=2,num_categories=number_of_breaks).to(device)
pi_network= PolicyNetwork(state_dim=2).to(device)

optimizer_qFunc = optim.Adam(Q_function.parameters(), lr=0.0003)  # Learning rate is set to 0.001
optimizer_vFunc = optim.Adam(V_function.parameters(), lr=0.0003)  # Learning rate is set to 0.001
optimizer_piNet = optim.Adam(pi_network.parameters(), lr=0.0001)  # Learning rate is set to 0.005
Expectile_Loss=ExpectileLoss()
Exponential_Loss=Exponential_loss()
gamma=0.95
mseloss = torch.nn.MSELoss()
batch_count=0
loss_values = {
    'Lv': [],
    'Lq': [],
    'Lpi': [],
}
Lv_save_name='Lv_save.npy'
Lq_save_name='Lq_save.npy'
Lpi_save_name='Lpi_save.npy'

Lv_min = torch.inf
Lq_min =torch.inf
Lpi_min = torch.inf

q_function_path = 'q_function_model.pth'
v_function_path = 'v_function_model.pth'
pi_network_path = 'pi_network_model.pth'
last_saved_epoch=0
for epoch in range(num_epochs):
    print(epoch," epoch is starting")
    for batch in tqdm(dataloader):
        state_array, action_array, state2_array, state_action = batch
       #print(jd)
        #print(state_array.shape, action_array.shape, state2_array.shape, state_action.shape)
        action_array = action_array.permute(0, 2, 1)*(-1)
        #print(state_array.shape, action_array.shape, state2_array.shape, state_action.shape)
        # x=torch.squeeze(state_array)
        # print(x.shape)
        # print(jd)
        
        #print(jd)

        state_action = state_action.to(device)
        output = best_model_trained(state_action).detach()
        expanded_tensor = output.unsqueeze(1).repeat(1, state_action.shape[1], 1)
        reward_array = 60*expanded_tensor.mean(dim=-1)
        # print(jd)
        # state_array =state_array.reshape(-1,3).to(device)
        # state2_array =state2_array.reshape(-1,3).to(device)
        # reward_array =reward_array.reshape(-1,1).to(device)
        # action_array =action_array.reshape(-1,1).to(device)
        
        state_array =state_array.to(device)
        state2_array =state2_array.to(device)
        reward_array =reward_array.to(device)
        action_array =action_array.to(device)


        # print("&&&&&&&&&&&&&&&&&&&&",state_array.shape, action_array.shape, 
        #       state2_array.shape, reward_array.shape)
        # print(jd)
        #print(torch.cat((state_array, action_array), dim=1).shape)
        
        #action_array=torch.unsqueeze(action_array,2)
        #print(state_array.shape, action_array.shape,"$$$$$$$")
        #print(jd)
        #print(torch.stack((state_array, action_array), dim=1).shape)
        #print(state_array.shape, action_array.shape, state2_array.shape, state_action.shape)
        q_value_input= torch.cat((state_array, action_array), dim=2).to(device)
        steps_1 = 1

        #print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        for i in range(steps_1):
           # print("#########################################################")
            q_function_output=Q_function(state_array,action_array)
            #print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            #print(jd)
            v_function_output=V_function(state_array)
            v2_function_output=V_function(state2_array)
            #print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            #print(q_function_output.shape,v_function_output.shape, "|||||||||||||||||" )
            #Lv= mseloss(q_function_output,torch.squeeze(v_function_output))##Represents the closeness of functions
            #Lv+=mseloss(reward_array,torch.squeeze(v_function_output))##Should represent the actual reward prediction capacity
            Lv=mseloss(reward_array,q_function_output)
            #Lv+=mseloss()
            #Lv/=2

            loss_values['Lv'].append(Lv.item())
            np.save(Lv_save_name, loss_values['Lv'])

            # optimizer_vFunc.zero_grad()
            # Lv.backward()
            # optimizer_vFunc.step()
            ################
            optimizer_qFunc.zero_grad()
            optimizer_vFunc.zero_grad()
            Lv.backward()
            optimizer_qFunc.step()
            optimizer_vFunc.step()
            ################

            # q_function_output=Q_function(state_array,action_array)
            # #print(q_function_output.shape,"$$$$$$$$$$$$$$$$$")
            # v_function_output=V_function(state_array)
            # v2_function_output=V_function(state2_array)
            # #print(reward_array.shape,v2_function_output.shape,q_function_output.shape,"###")
            # Lq= mseloss(reward_array+gamma*torch.squeeze(v2_function_output),q_function_output)
            # loss_values['Lq'].append(Lq.item())
            # np.save(Lq_save_name, loss_values['Lq'])
            #print(Lq.shape,"Loss from Q function gradinet")
            # optimizer_qFunc.zero_grad()
            # Lq.backward()
            # optimizer_qFunc.step()

            #print(Lv.item(),Lq.item())
            # if(Lq<Lq_min):
            #     Lq_min=Lq
            #     torch.save(Q_function.state_dict(), q_function_path)
            #     last_saved_epoch=epoch
            if(Lv<Lv_min):
                Lv_min=Lv
                torch.save(V_function.state_dict(), v_function_path)
                last_saved_epoch=epoch



        steps_2=0
        for i in range(steps_2):
            optimizer_piNet.zero_grad()
            #print(state_array.shape, action_array.shape," Pi function input shape")
            #print(state_array.shape,action_array.shape,"!")
            #print(action_array)
            mu1,mu2=pi_network(state_array)
            #mu,sigma =
            average_mu =(mu1+mu2)/2
            stacked_tensor = torch.stack([mu1, mu2], dim=-1)
            # Calculate the standard deviation along the last dimension
            #print(stacked_tensor.shape,"&&&&&&")
            std_tensor = torch.std(stacked_tensor, dim=-1, keepdim=True)

            #print(std_tensor.shape,"&&&&&&")
            #print(average_mu.shape,std_tensor.shape,"###")
            normal_dist = Normal(average_mu.unsqueeze(-1), std_tensor)

            log_prob = normal_dist.log_prob(action_array)
            log_prob = torch.diagonal(log_prob)
            #print(mu.shape,sigma.shape,log_prob.shape,")(((((((((s)))))))))")
            #print(jd)
            q_function_output=Q_function(state_array,action_array)
            v_function_output=V_function(state_array)
            #print(log_pi_function_output.shape,"log pi outputs")
            #print(q_function_output.shape,v_function_output.shape,"JJJDJDJDJDJDJD")
            L_pi= -log_prob#+3
            #print(q_function_output.shape,v_function_output.shape)
            L_pi =torch.mean(L_pi)*Exponential_Loss(q_function_output,torch.squeeze(v_function_output))


            #print(L_pi.shape,"Loss from policy gradinet")
            # print(L_pi)
            # print(jd)
            L_pi= torch.mean(L_pi)
            loss_values['Lpi'].append(L_pi.item())
            np.save(Lpi_save_name, loss_values['Lpi'])
            
            #print(jd)
            L_pi.backward()
            optimizer_qFunc.step()
            #print(gjsih)
            if(L_pi<Lpi_min):
                Lpi_min=L_pi
                torch.save(pi_network.state_dict(), pi_network_path)
                last_saved_epoch=epoch
    print(f'Last saved epoch was {last_saved_epoch}')

        
        # if(batch_count==10):
        #     break
        
