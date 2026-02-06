import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F

class DataConverter():
    def __init__(self):
        pass
    
    # ----- Loading From Path -----
    def load_path_as_numpy(self, path: str):
        """
        Function to load a NIfTI volume as a numpy array using the data path
        """
        img = nib.load(path)
        data = img.get_fdata()
        return data
    
    def load_path_as_tensor(self, path: str, device = 'cuda'):
        """
        Function to load a NIfTI volume as a torch tensor using the data path
        """
        data = self.load_path_as_numpy(path)
        tensor = torch.tensor(data, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # Get correct (1, 1, D, H, W) structure
        return tensor.float()
    
    def get_middle_slice_from_path(self, data_path: str):
        data = self.load_path_as_numpy(data_path)
        return data[:, :, data.shape[2] // 2]

    def get_slice_from_path(self, data_path: str, index: int = 0):
        data = self.load_path_as_numpy(data_path)
        return data[:, :, index]
    
    def get_all_slices_as_tensor(self, data_path:str, device = 'cuda'):
        data = self.load_path_as_numpy(data_path)
        return [torch.tensor(slice, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0) for slice in data.transpose(2, 0, 1)]

    # ----- Convert between Numpy and Tensor -----
    def numpy_to_tensor(self, array: np.ndarray, device: str = 'cuda'):
        tensor = torch.tensor(array, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # Get correct (1, 1, D, H, W) structure
        return tensor.float()
    
    def tensor_to_numpy(self, tensor: torch.Tensor):
        # TODO Sjekk at funker
        return tensor.detach().cpu().numpy()
    
    # TODO Make not hardcoded
    def get_patient_feature_vector(self, data_path: str, usage_list=None) -> torch.Tensor:
        usage_list = usage_list or [False, False, False, False, False, False]
        vectors = torch.tensor([0.0, 1.0, 0.5, 1.0, 0.25, 0.75], dtype=torch.float32)

        mask = torch.tensor(usage_list, dtype=torch.float32)
        feature_vector = vectors * mask
        return feature_vector.unsqueeze(0)  # [1, 6]
    
    # ----- Volume Size Changes -----
    def get_volume_with_3d_change(
        self,
        tensor: torch.Tensor,
        crop_axes: tuple[tuple[int, int, int], tuple[int, int, int]],
        remove_mode: bool = True,
    ):
        """
        tensor: (B, C, D, H, W)
        crop_axes:
            ((d_start, h_start, w_start),
            (d_end,   h_end,   w_end))
        remove_mode:
            True  -> remove margins
            False -> pad with zeros (black)
        """

        d0, h0, w0 = crop_axes[0]
        d1, h1, w1 = crop_axes[1]

        if remove_mode:
            D, H, W = tensor.shape[-3:]
            return tensor[:, :, d0:D - d1, h0:H - h1, w0:W - w1]

        else:
            # F.pad expects padding in reverse order:
            # (w_left, w_right, h_left, h_right, d_left, d_right)
            padding = (w0, w1, h0, h1, d0, d1)
            return F.pad(tensor, padding, mode="constant", value=0)