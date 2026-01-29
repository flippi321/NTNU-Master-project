from utils.data_loader import DataLoader
from utils.data_converter import DataConverter
from utils.data_analyser import DataAnalyser
import os

print("Starting...")

data_loader = DataLoader()
data_converter = DataConverter()
data_analyser = DataAnalyser()

print("Loader, converter and analyser ready...")

hunt_3_path, hunt_4_path = data_loader.get_random_pair_path()
h3_num, h4_num = data_converter.load_path_as_numpy(hunt_3_path), data_converter.load_path_as_numpy(hunt_4_path)

h3_tensor = data_converter.numpy_to_tensor(h3_num, 'cpu')
h4_tensor = data_converter.numpy_to_tensor(h4_num, 'cpu')


data = data_analyser.get_data_info(data_loader, data_converter, max_layers=8)

try:
    file_path = "file-size-reccomendations.txt"
    # Remove old recommendations
    if os.path.exists(file_path):
        os.remove(file_path)

    # Add recommendations
    with open(file_path, "w") as file_object:
        for i, (crops, rec) in enumerate(zip(data[-1], data[-2])):
            text = f"{i+1} layers:   {rec}      |   Crops required:    {crops}\n"
            file_object.write(text)
    print(f"Done!")
except IOError as e:
    print(f"Could not write to file :/")