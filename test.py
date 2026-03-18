from utils.data_loader import DataLoader
from utils.data_converter import DataConverter
from utils.data_analyser import DataAnalyser
from utils.loss_functions import ssim_loss
from utils.train_eval import fit_feature_based_3D
from models.unet_3d_std import UNet3D
import os
import torch

print("Starting...")

# Initialize utilities
data_loader = DataLoader()
data_converter = DataConverter()
data_analyser = DataAnalyser()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_set, validation_set, test_set = data_loader.split_dataset_paths(seed=42)
output_dir = "out/unets"
unet_path = f"{output_dir}/3d_unet_model_standard.pt"
unet = UNet3D(in_ch=1, base=32).to(device)
crop_axes=((16, 10, 0), (17, 11, 17))
n_gen = 5

"""
unet, history, snapshots, best = fit_feature_based_3D(
    model=unet,
    device=device,
    training_pairs=test_set,
    validation_pairs=validation_set,
    epochs=2500,
    dataConverter=data_converter,
    snapshot_every=500,
    checkpoint_every=250,
    loss_func=ssim_loss,
    crop_axes=((16, 10, 0), (17, 11, 17)),
    feature_usage_list=[True, True, True, True, True, True],
)
"""

# Load existing model if it exists
if os.path.isfile(unet_path):
    print("Loading existing model")
    unet.load_state_dict(torch.load(unet_path))
else:
    print("No existing 3D U-Net found, starting fresh training.")
    os.makedirs(output_dir, exist_ok=True)

os.makedirs("out/gen/x", exist_ok=True)
os.makedirs("out/gen/y", exist_ok=True)
os.makedirs("out/gen/y_hat", exist_ok=True)

# Get random pair
for i in range(n_gen):
    x_pth, y_pth = data_loader.get_random_pair_path()

    x_whole = data_converter.load_path_as_tensor(x_pth, device)
    y_whole = data_converter.load_path_as_tensor(y_pth, device)

    x = data_converter.get_volume_with_3d_change(x_whole, crop_axes)

    y_hat = unet(x)
    y_hat_whole = data_converter.get_volume_with_3d_change(y_hat, crop_axes, remove_mode=False)

    x_num       = data_converter.tensor_to_numpy(x_whole)
    y_num       = data_converter.tensor_to_numpy(y_whole)
    y_hat_num   = data_converter.tensor_to_numpy(y_hat_whole)

    data_converter.numpy_to_nib_gz(x_num, f"out/gen/x/{i}")
    data_converter.numpy_to_nib_gz(y_num, f"out/gen/y/{i}")
    data_converter.numpy_to_nib_gz(y_hat_num, f"out/gen/y_hat/{i}")

print("Done!")