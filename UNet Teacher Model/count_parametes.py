import torch
from unet_model import UNet


def count_parameters(model):
    # Total parameters (including those frozen)
    total_params = sum(p.numel() for p in model.parameters())
    # Parameters that will be updated during training
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    # Estimate Flash size for INT8 (1 byte per parameter)
    print(f"Estimated INT8 Flash Size: {total_params / 1024:.2f} KB")
    return total_params


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = UNet(in_channels=1, base_filters=48).to(device)

# 2. Load your weights
weights_path = "results/teacher/teacher_unet.pth"
try:
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"Successfully loaded: {weights_path}")
except FileNotFoundError:
    print("Weight file not found, counting initialized parameters instead.")

# 3. Run the count
count_parameters(model)
