import os
import random
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
import numpy as np

class DataAnalyser():
    """
    Heads up, this file is messy, hardcoded and not really optimized.

    Should work fine tho :P
    """
    
    def __init__(self):
        pass

    def ssim(self, hunt3, hunt4):
        ssim, _ = structural_similarity(hunt3, hunt4, data_range=1, channel_axis=None, full=True)
        return ssim

    def get_data_info(self, max_entries:int=None):
        """
        Function to print the number of entries, average value for entries and the dimensions of the dataset
        """
        # Get number of entries in each hunt dataset
        hunt3_num = len(os.listdir(os.path.join(self.hunt_path, self.hunts[0])))
        hunt4_num = len(os.listdir(os.path.join(self.hunt_path, self.hunts[1])))
        print(f"Number of entries in {self.hunts[0]}: {hunt3_num}")
        print(f"Number of entries in {self.hunts[1]}: {hunt4_num}")

        # For every candidate we get the MRI pair data
        means_h3 = []
        min_h3_shape = min_h4_shape = [np.inf, np.inf, np.inf]
        max_h3_shape = max_h4_shape = [0, 0, 0]
        means_h4 = []
        for i, candidate in enumerate(os.listdir(os.path.join(self.hunt_path, self.hunts[0]))):
            
            # We load the data
            hunt3 = self.load_from_path(self.get_pair_path_from_id(candidate)[0])
            hunt4 = self.load_from_path(self.get_pair_path_from_id(candidate)[1])

            # Get average
            means_h3.append(np.mean(hunt3))
            means_h4.append(np.mean(hunt4))

            # Get min and max shape
            min_h3_shape = [min(min_h3_shape[0], hunt3.shape[0]), min(min_h3_shape[1], hunt3.shape[1]), min(min_h3_shape[2], hunt3.shape[2])]
            max_h3_shape = [max(max_h3_shape[0], hunt3.shape[0]), max(max_h3_shape[1], hunt3.shape[1]), max(max_h3_shape[2], hunt3.shape[2])]
            min_h4_shape = [min(min_h4_shape[0], hunt4.shape[0]), min(min_h4_shape[1], hunt4.shape[1]), min(min_h4_shape[2], hunt4.shape[2])]
            max_h4_shape = [max(max_h4_shape[0], hunt4.shape[0]), max(max_h4_shape[1], hunt4.shape[1]), max(max_h4_shape[2], hunt4.shape[2])]

            if(max_entries and i >= max_entries):
                break
        
        # Print Average intensity and shape info
        hunt3_mean = np.mean(means_h3)
        hunt4_mean = np.mean(means_h4)
        print(f"Average intensity across Hunt3: {hunt3_mean}")
        print(f"Average intensity across Hunt4: {hunt4_mean}")

        print(f"Min shape across Hunt3: {min_h3_shape}, Max shape across Hunt3: {max_h3_shape}")
        print(f"Min shape across Hunt4: {min_h4_shape}, Max shape across Hunt4: {max_h4_shape}")

        return hunt3_num, hunt4_num, hunt3_mean, hunt4_mean, min_h3_shape, max_h3_shape, min_h4_shape, max_h4_shape
    
    def display_slices(self, slices, slice_labels, slice_colors=None):
        """
        function to display multiple slices side by side for comparison
        """
        # Create figure with 1 row and 2 columns
        fig, axs = plt.subplots(1, len(slices), figsize=(10, 5))

        # Show HUNT3 image
        for i, slice in enumerate(slices):
            axs[i].imshow(slice, cmap='gray' if not slice_colors else slice_colors[i])
            axs[i].set_title(slice_labels[i])
            axs[i].axis('off')

        plt.tight_layout()
        plt.show()

    def display_slice_differences(self, slice1, slice2, hot=False):
        diff_slice = np.abs(slice1 - slice2)
        plt.figure(figsize=(6, 6))
        plt.axis('off')

        # Display only the differences
        if(hot):
            plt.imshow(diff_slice, cmap='hot')
            plt.title('Differences between HUNT3 and HUNT4')
            plt.colorbar(label='Difference Intensity')

        # Display everything, with differences colored
        else:
            plt.imshow(slice1, cmap='gray')
            plt.imshow(diff_slice, alpha=0.5)
            plt.title('HUNT3 slice with HUNT3-HUNT4 differences highlighted')
        
        plt.show()