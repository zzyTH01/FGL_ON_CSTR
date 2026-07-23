import pickle
import torch
from torch.utils.data import Dataset
from utils import MackeyGlass
# Assuming the MackeyGlass class and dependencies (jitcdde_lyap, numpy) are available

# Define the parameters for the Mackey-Glass series
mg_params = {
    'tau': 17,
    'constant_past': 0.9,
    'dt': 1.0,
    'splits': (10000., 0.), 
    'seed_id': 42
}

# Instantiate the class to generate the data
print("Generating Mackey-Glass time series...")
mg_dataset = MackeyGlass(**mg_params)

time_series_list = []

for idx in range(len(mg_dataset)):
    _, target = mg_dataset[idx] 
    time_series_list.append(target.squeeze().item())

series_column_1 = torch.tensor(time_series_list, dtype=torch.float64).unsqueeze(1)

series_column_2 = series_column_1.clone()

time_series_data = torch.cat((series_column_1, series_column_2), dim=1)

print(f"Generation complete. Data shape: {time_series_data.shape}")

output_filename = "data.pkl"

with open(output_filename, 'wb') as f:
    pickle.dump(time_series_data, f)

print(f"Successfully saved the two-column time series to '{output_filename}'")

with open(output_filename, 'rb') as f:
    loaded_data = pickle.load(f)
print(f"Verification: Loaded data shape is {loaded_data.shape}")
print(f"Verification: First few points of Column 1: {loaded_data[:5, 0].tolist()}")
print(f"Verification: First few points of Column 2: {loaded_data[:5, 1].tolist()}")
