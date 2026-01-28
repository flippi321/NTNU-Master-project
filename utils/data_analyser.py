import os
import random
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
from tqdm import tqdm
import numpy as np

class DataAnalyser():
    """
    Heads up, this file is messy, hardcoded and not really optimized.

    Should work fine tho :P
    """
    
    def __init__(self, root_path: str = '/cluster/projects/vc/data/mic/closed/MRI_HUNT/images/images_3D_preprocessed/'):
        self.datasets = ['HUNT3', 'HUNT4']
        self.root = root_path
        self.all_candidates = os.listdir(os.path.join(self.root, self.datasets[0]))
        pass

    def ssim(self, hunt3, hunt4):
        ssim, _ = structural_similarity(hunt3, hunt4, data_range=1, channel_axis=None, full=True)
        return ssim

    def get_data_info(self, data_loader, data_converter, max_entries: int = None, eps: float = 1e-6, label_w = 18):
        """
        Function to print the number of entries, average value for entries and the dimensions of the dataset
        """
        # Get number of entries in each hunt dataset
        hunt3_num = len(os.listdir(os.path.join(self.root, self.datasets[0])))
        hunt4_num = len(os.listdir(os.path.join(self.root, self.datasets[1])))

        # For every candidate we get the MRI pair data
        means_h3, means_h4 = [], []
        min_h3_shape = min_h4_shape = [np.inf, np.inf, np.inf]
        max_h3_shape = max_h4_shape = [0, 0, 0]

        # NEW: per-face crop stats (x_str, x_end, y_str, y_end, z_str, z_end)
        face_names = ["x_str", "x_end", "y_str", "y_end", "z_str", "z_end"]

        min_h3_face = np.array([np.inf]*6, dtype=float)
        max_h3_face = np.array([0]*6, dtype=int)
        min_h4_face = np.array([np.inf]*6, dtype=float)
        max_h4_face = np.array([0]*6, dtype=int)

        for i, candidate in tqdm(enumerate(self.all_candidates), total=hunt3_num, desc="Analyzing candidates", leave=False):
            candidate_path_pairs = data_loader.get_pair_path_from_id(candidate)
            hunt3 = data_converter.load_path_as_numpy(candidate_path_pairs[0])
            hunt4 = data_converter.load_path_as_numpy(candidate_path_pairs[1])

            # Get average
            means_h3.append(np.mean(hunt3))
            means_h4.append(np.mean(hunt4))

            # Get min and max shape
            min_h3_shape = [min(min_h3_shape[0], hunt3.shape[0]), min(min_h3_shape[1], hunt3.shape[1]), min(min_h3_shape[2], hunt3.shape[2])]
            max_h3_shape = [max(max_h3_shape[0], hunt3.shape[0]), max(max_h3_shape[1], hunt3.shape[1]), max(max_h3_shape[2], hunt3.shape[2])]
            min_h4_shape = [min(min_h4_shape[0], hunt4.shape[0]), min(min_h4_shape[1], hunt4.shape[1]), min(min_h4_shape[2], hunt4.shape[2])]
            max_h4_shape = [max(max_h4_shape[0], hunt4.shape[0]), max(max_h4_shape[1], hunt4.shape[1]), max(max_h4_shape[2], hunt4.shape[2])]

            # HUNT3 per-face crop
            mins3, maxs3, size3, faces3 = hunt_per_face_possible_crop(hunt3, eps=eps)
            
            # The axis changes for the HUNT 3 volume
            faces3 = np.array(faces3, dtype=int)
            min_h3_face = np.minimum(min_h3_face, faces3)
            max_h3_face = np.maximum(max_h3_face, faces3)


            # The axis changes for HUNT 4 volume
            mins4, maxs4, size4, faces4 = hunt_per_face_possible_crop(hunt4, eps=eps)
            faces4 = np.array(faces4, dtype=int)
            min_h4_face = np.minimum(min_h4_face, faces4)
            max_h4_face = np.maximum(max_h4_face, faces4)

            if (max_entries and i > max_entries):
                break

        # Calculate Average intensity and shape info
        hunt3_mean = np.mean(means_h3)
        hunt4_mean = np.mean(means_h4)

        # Print Average intensity and shape info
        print(f"\n---------------  HUNT 3  ---------------")
        print(f"{'Number of entries':>{label_w}} : {hunt3_num}")
        print(f"{'Average intensity':>{label_w}} : {hunt3_mean:.5f}")
        print(f"{'Min shape':>{label_w}} : {min_h3_shape}")
        print(f"{'Max shape':>{label_w}} : {max_h3_shape}")
        _, _, reccomended_dim_h3 = print_face_summary(min_h3_face, min_h3_shape, max_h3_face, face_names, label_w)

        print(f"\n---------------  HUNT 4  ---------------")
        print(f"{'Number of entries':>{label_w}} : {hunt4_num}")
        print(f"{'Average intensity':>{label_w}} : {hunt4_mean:.5f}")
        print(f"{'Min shape':>{label_w}} : {min_h4_shape}")
        print(f"{'Max shape':>{label_w}} : {max_h4_shape}")
        _, _, reccomended_dim_h4 = print_face_summary(min_h4_face, min_h4_shape, max_h4_face, face_names, label_w)

        print(f"\n-----------------  DIV ------------------")
        recommended_dim = [int(max(a, b)) for a, b in zip(reccomended_dim_h3, reccomended_dim_h4)]
        print(f"{'Recommended dim':>{label_w}} : {recommended_dim}")

     
        return hunt3_num, hunt4_num, hunt3_mean, hunt4_mean, min_h3_shape, max_h3_shape, min_h4_shape, max_h4_shape, min_h3_face, max_h3_face, min_h4_shape, max_h4_face, recommended_dim
    
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

# -----------------------------------------------------
#                   HELPER FUNCTIONS
# -----------------------------------------------------
def hunt_per_face_possible_crop(hunt_vol: np.ndarray, eps: float = 1e-6):
    """
    For every dimention/axis, we see how much we can crop before we hit the brain
    This is done both directions per axis, as the padding isn't symetrical (we use Start and End)

    Returns:
      mins: (x0, y0, z0) inclusive
      maxs: (x1, y1, z1) exclusive
      size: (dx, dy, dz)
      crop_faces: (x_str, x_end, y_str, y_end, z_str, z_end)
    """
    mask = hunt_vol > eps

    shape = np.array(hunt_vol.shape, dtype=int)
    mins = np.zeros(3, dtype=int)
    maxs = shape.copy()

    # crop faces in order: x_str, x_end, y_str, y_end, z_str, z_end
    crop_faces = np.zeros(6, dtype=int)

    for axis in range(3):
        other_axes = tuple(ax for ax in range(3) if ax != axis)

        # line[i] is True if slice i along this axis contains ANY foreground voxel
        line = mask.any(axis=other_axes)

        # first/last slice with any foreground
        first = int(np.argmax(line))
        last = int(shape[axis] - 1 - np.argmax(line[::-1]))

        start_crop = first
        end_crop = (shape[axis] - 1) - last

        mins[axis] = start_crop
        maxs[axis] = shape[axis] - end_crop

        crop_faces[2*axis + 0] = start_crop
        crop_faces[2*axis + 1] = end_crop

    size = maxs - mins
    return tuple(mins), tuple(maxs), tuple(size), tuple(crop_faces)

def print_face_summary(min_face, min_shape, max_face, face_names, label_w):
    """
    Prints crop ranges + resulting size ranges.
    Returns:
      result_min_xyz: smallest possible resulting size after cropping (x,y,z)
      result_max_xyz: largest  possible resulting size after cropping (x,y,z)
      recommended_xyz: next divisible-by-8 size >= result_min_xyz
    """
    min_face = min_face.astype(int)
    max_face = max_face.astype(int)
    min_shape = np.array(min_shape, dtype=int)

    # Split faces into start/end per axis
    start_min = np.array([min_face[0], min_face[2], min_face[4]])
    start_max = np.array([max_face[0], max_face[2], max_face[4]])
    end_min   = np.array([min_face[1], min_face[3], min_face[5]])
    end_max   = np.array([max_face[1], max_face[3], max_face[5]])

    # Crop ranges (per axis)
    crop_min = start_min + end_min  # smallest crop
    crop_max = start_max + end_max  # largest crop

    # Resulting size ranges:
    result_min = min_shape - crop_max
    result_max = min_shape - crop_min

    # Recommended CNN size: next multiple of 8 >= smallest resulting size
    recommended = next_divisible_by_8(result_min)

    start_fmt = [f"{start_min[i]}-{start_max[i]}" for i in range(3)]
    end_fmt   = [f"{end_min[i]}-{end_max[i]}"     for i in range(3)]
    print(f"{'Start crops':>{label_w}} : [{', '.join(start_fmt)}]")
    print(f"{'End crops':>{label_w}} : [{', '.join(end_fmt)}]")

    # Show resulting size range as [min-max] per axis
    size_fmt = [f"{result_min[i]}-{result_max[i]}" for i in range(3)]
    print(f"{'Resulting size':>{label_w}} : [{', '.join(size_fmt)}]")
    return result_min, result_max, recommended

def next_divisible_by_8(dims_xyz):
    """
    Returns the smallest dims (x,y,z) where each is divisible by 8 AND >= input.
    Works for list/tuple/np.array of length 3.
    """
    dims = np.array(dims_xyz, dtype=int)
    return ((dims + 7) // 8) * 8
