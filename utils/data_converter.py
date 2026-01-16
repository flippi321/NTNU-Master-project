import torch
import numpy as np
import nibabel as nib

class DataConverter():
    def __init__(self):
        pass
    
    # ----- Loading From Path -----
    def load_path_as_numpy(self, path):
        """
        Function to load a NIfTI volume as a numpy array using the data path
        """
        img = nib.load(path)
        data = img.get_fdata()
        return data
    
    def load_path_as_tensor(self, path):
        """
        Function to load a NIfTI volume as a torch tensor using the data path
        """
        data = self.load_path_as_numpy(path)
        tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # TODO: Sjekk om vi kan fjerne unsqueeze
        return tensor.float()
    
    def get_middle_slice_from_path(self, data_path):
        data = self.load_path_as_numpy(data_path)
        return data[:, :, data.shape[2] // 2]

    def get_slice_from_path(self, data_path, index=0):
        data = self.load_path_as_numpy(data_path)
        return data[:, :, index]
    
    def get_all_slices_as_tensor(self, data_path):
        data = self.load_path_as_numpy(data_path)
        return [torch.tensor(slice, dtype=torch.float32) for slice in data.transpose(2, 0, 1)]

    # ----- Convert between Numpy and Tensor -----
    def numpy_to_tensor(self, array: np.ndarray):
        tensor = torch.tensor(array, dtype=torch.float32).unsqueeze(0)  # TODO: Sjekk om vi kan fjerne unsqueeze
        return tensor.float()
    
    def tensor_to_numpy(self, tensor: torch.Tensor):
        # TODO Sjekk at funker
        return tensor.detach().cpu().numpy()