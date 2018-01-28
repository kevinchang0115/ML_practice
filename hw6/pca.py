import os
import sys
import numpy as np
from skimage import io

photo_dir = sys.argv[1]
photo_name = sys.argv[2]
photo_path = os.path.join(photo_dir, photo_name)

# Read all image
img_all = []
for photo in sorted(os.listdir(photo_dir)):
    if photo.endswith('.jpg'):
        img = io.imread(os.path.join(photo_dir, photo))
        img_all.append(img.flatten())
img_all = np.array(img_all)

# SVD
img_all = img_all / 255.0
mean = np.mean(img_all, axis=0)
div = img_all - mean
U, s, V = np.linalg.svd(div.T, full_matrices=False)

# Read target img
img = io.imread(photo_path)
size = img.shape
img_data = img.flatten()
img_data = img_data / 255.0

# Reconstruct
eigen_num = 4
recon = img_data-mean 
weight = np.dot(recon,U[:,:eigen_num])
recon = np.dot(U[:,:eigen_num],weight.T)
recon += mean
recon -= np.min(recon)
recon /= np.max(recon)
recon = (recon * 255).astype(np.uint8)
io.imsave('reconstruction.jpg', recon.reshape(size))
