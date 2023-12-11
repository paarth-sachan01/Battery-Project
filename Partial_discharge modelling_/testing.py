import torch
#import torchvision.models as models

# Step 1: Create an instance of the model with the same architecture
model = models.resnet18(pretrained=False)  # Replace with the actual model class and its configuration.

# Step 2: Load weights from .ckpt files
checkpoint_path = '/Users/paarthsachan/technical/State_of_health_battery/Partial_discharge_modelling/checkpoints/best_model-v1.ckpt'  # or 'path_to_checkpoint_file.pth'
checkpoint = torch.load(checkpoint_path)

# Load the model state_dict (weights) from the checkpoint
model.load_state_dict(checkpoint['model_state_dict'])

# If you need to load other items like optimizer state or epoch information:
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']

model.eval()  # Set the model in evaluation mode (if necessary)

# Now, you can use the model for inference or further training.
