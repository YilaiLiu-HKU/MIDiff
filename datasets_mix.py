import numpy as np
origin_data_path='/home/yilai/poster/NetDiffus/dataset/all_users_data_with6cluster.npz'
generated_data_path='/home/yilai/poster/NetDiffus/recovered_traces_MEAN1.5.npz'
output_path='/home/yilai/poster/NetDiffus/mix_datasets/origin.npz'
origin_file=np.load(origin_data_path)
app_traffics=origin_file['Category_ID_Traffic (Byte)']
pois=origin_file["poi_labels"]
generated_file=np.load(generated_data_path)
generated_app_traffics=generated_file['app_traces']
generated_pois=generated_file["poi_traces"]
MEAN_FACTOR=generated_file['MEAN_FACTOR']
app_traffics=np.concat((app_traffics,generated_app_traffics),axis=0)
pois=np.concat((pois,generated_pois),axis=0)
np.savez(
    output_path,
    app_traces=np.array(app_traffics),
    poi_traces=np.array(pois),
    MEAN_FACTOR=MEAN_FACTOR,
)
print(f"Saved to {output_path}")