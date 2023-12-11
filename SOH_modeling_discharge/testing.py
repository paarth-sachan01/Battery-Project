import torch

# Create a sample PyTorch tensor
tensor = torch.tensor([1, 2, 3, 4, 5])

# Define a list of indices you want to extract
indices_to_extract = [0, 2, 4]

# Use torch.index_select to extract elements
selected_elements = torch.index_select(tensor, dim=0, index=torch.tensor(indices_to_extract))

# Print the selected elements
print(selected_elements)
