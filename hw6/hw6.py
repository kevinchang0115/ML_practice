import sys
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.cluster import KMeans

# Load encoder
encoder = load_model("model/encoder.h5")

# Load imgs
X = np.load(sys.argv[1])
X = X.astype('float32') / 255.
X = X.reshape((len(X), -1))

# Cluster
encoded_imgs = encoder.predict(X)
encoded_imgs = encoded_imgs.reshape(encoded_imgs.shape[0], -1)
kmeans = KMeans(n_clusters=2, random_state=0).fit(encoded_imgs)

# Read test case
f = pd.read_csv(sys.argv[2])
IDs, idx1, idx2 = np.array(f['ID']), np.array(f['image1_index']), np.array(f['image2_index'])

# Predict
o = open(sys.argv[3], 'w')
o.write("ID,Ans\n")
for idx, i1, i2 in zip(IDs, idx1, idx2):
    p1 = kmeans.labels_[i1]
    p2 = kmeans.labels_[i2]
    if p1 == p2:
        pred = 1
    else: 
        pred = 0
    o.write("{},{}\n".format(idx, pred))
o.close()