import pandas as pd
import numpy as np
import os
from pyts.image import GramianAngularField
import matplotlib.pyplot as plt
from skimage.transform import resize

data_dir = "Youtube/vid"  # directory where 1D traces are stored (Format : Youtube/vid*)
save_dir = "Youtube_img/vid"  # directory to store GASF converted data as PNG
n = 20  # number of videos

for i in range(n):
    vid = data_dir + str(i + 1)
    path_to_save = os.path.abspath(save_dir + str(i + 1))
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    k = 1
    for file in os.scandir(vid):  # load data and convert to GASF
        df = pd.read_csv(file, usecols=[' addr2_bytes'])  # the column to read the data
        df = df.replace(np.nan, 0)
        data = df.to_numpy(dtype='float')
        pts = []
        # GASF formation
        for l in data:
            for j in l:
                pts.append(j)
        num_of_samples_per_bin = 4  # to bin the video data
        points = []
        for j in range(int(len(pts) / num_of_samples_per_bin)):
            points.append(np.sum(pts[j * num_of_samples_per_bin:(j + 1) * num_of_samples_per_bin]))
        X = np.array([points])
        # Compute Gramian angular fields
        gasf = GramianAngularField(sample_range=(0, 1), method='summation')
        X_gasf = gasf.fit_transform(X)
        # Gamma correction
        gasf_img = X_gasf[0] * 0.5 + 0.5
        gamma = 0.25
        gasf_img = np.power(gasf_img, gamma)

        # Resize the image to 128x128
        gasf_img_resized = resize(gasf_img, (128, 128), anti_aliasing=True)

        # Save as PNG
        plt.imsave(os.path.join(path_to_save, f"vid{i + 1}_{k}.png"), gasf_img_resized, cmap='viridis')

        k += 1