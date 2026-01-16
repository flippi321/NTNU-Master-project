from utils.data_loader import DataLoader
from utils.data_converter import DataConverter

data_loader = DataLoader()
data_converter = DataConverter()

hunt_3_path, hunt_4_path = data_loader.get_random_pair_path()
h3_num, h4_num = data_converter.load_path_as_numpy(hunt_3_path), data_converter.load_path_as_numpy(hunt_4_path)

h3_tensor = data_converter.numpy_to_tensor(h3_num)
h4_tensor = data_converter.numpy_to_tensor(h4_num)

print(h3_tensor.shape, data_converter.to_torch_img(h3_num, 'cuda').shape)