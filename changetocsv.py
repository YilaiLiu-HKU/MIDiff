import numpy as np
import pandas as pd

# Assuming you've loaded your data like this:
data = np.load('/data/yilai/MiDiff/ckpt/ckpt/tiff_log_attnBoth/48500.npz')
app_traces = data['app_traces']  # Shape: B x time_step x num_apps
poi_traces = data['poi_traces']  # Shape: B x time_step x num_pois
file_path='attnboth.csv'
# Function to convert one-hot encoded traces to CSV format
def convert_to_csv(app_traces, poi_traces):
    rows = []
    
    # For each sample (B dimension)
    for sample_id in range(app_traces.shape[0]):
        # For each time step
        for time_step in range(app_traces.shape[1]):
            # Get app data (one-hot encoded)
            app_one_hot = app_traces[sample_id, time_step]
            
            # Convert one-hot to app_category and app_flow

            if np.sum(app_one_hot) == 0:
                # No app activity
                app_category = 0
                app_flow = 0.0
                poi_category = 0
            else:
                # Get the active app (assuming one-hot encoding)
                app_category = np.argmax(app_one_hot)  # +1 if your app IDs start from 1
                app_flow = np.max(app_one_hot)
            
            # Get poi data (one-hot encoded)
            poi_one_hot = poi_traces[sample_id, time_step]
            #import pdb;pdb.set_trace()
            if np.sum(poi_one_hot) == 0:
                # No poi activity
                poi_category = 0
                app_flow=0
                app_category=0

            else:
                # Get the active poi (assuming one-hot encoding)
                poi_category = np.argmax(poi_one_hot) 
            if poi_category==0:
                poi_category = 0
                app_flow=0
                app_category=0
            
            # Create row
            row = {
                'sample_id': sample_id,
                'time_step': time_step,
                'app_category': app_category,
                'app_flow': app_flow,
                'poi_category': poi_category
            }
            rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Reorder columns to match your format
    df = df[['sample_id', 'time_step', 'app_category', 'app_flow', 'poi_category']]
    
    return df

# Convert the data
csv_data = convert_to_csv(app_traces, poi_traces)

# Save to CSV
csv_data.to_csv(file_path, index=False)
print(f"转换完成，结果已保存到 {file_path}")