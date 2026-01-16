import os
import random

class DataLoader():
    def __init__(self, root_path: str = '/cluster/projects/vc/data/mic/closed/MRI_HUNT/images/images_3D_preprocessed/'):
        self.datasets = ['HUNT3', 'HUNT4']  # This won't change, so hardcoded it
        self.root = root_path
        self.all_candidates = os.listdir(os.path.join(self.root, self.datasets[0]))
        pass

    def get_pair_path_from_id(self, candidate:str):
        """
        Returns the Hunt3 and Hunt4 paths for a given candidate id
        """
        hunt3_path = os.path.join(self.root, self.datasets[0], candidate, f'{candidate}_0_T1_PREP_MNI.nii.gz')
        hunt4_path = os.path.join(self.root, self.datasets[1], candidate, f'{candidate}_1_T1_PREP_MNI.nii.gz')
        return hunt3_path, hunt4_path

    def get_random_pair_path(self):
        """
        Gets a random Hunt3 and Hunt4 pair path
        """
        random_candidate = random.choice(self.all_candidates)

        return self.get_pair_path_from_id(random_candidate)

    def get_all_pair_paths(self):
        """
        Function to get all the Hunt3 and Hunt4 pairs paths
        """
        all_pairs = [self.get_pair_path_from_id(candidate) for candidate in self.all_candidates]
        return all_pairs
    
    def split_dataset_paths(self, train_split:float = 0.70, val_split:float = 0.15, seed:int = random.randint(0, 10000)):
        """
        Function to split the dataset into training and testing paths
        """

        # We first shuffle all entries
        random.seed(seed)
        random.shuffle(self.all_candidates)

        # Get split index/points
        train_split_index = int(len(self.all_candidates) * train_split)
        val_split_index = int(len(self.all_candidates) * (train_split + val_split))
        
        # Split into train, test and eval
        train_entries = self.all_candidates[:train_split_index]
        val_entries = self.all_candidates[train_split_index:val_split_index]
        test_entries = self.all_candidates[val_split_index:]    # Test split is (1 - train - val)

        train_paths = [self.get_pair_path_from_id(candidate) for candidate in train_entries]
        val_paths = [self.get_pair_path_from_id(candidate) for candidate in val_entries]
        test_paths = [self.get_pair_path_from_id(candidate) for candidate in test_entries]

        return train_paths, val_paths, test_paths
