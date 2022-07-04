from scipy import sparse
from sklearn.decomposition import TruncatedSVD
import numpy as np


abs_path = "../"
# Load the saved co-occurence matrix
# Consider restarting the server if you are running the cell from training section as it might crash the server
co_occ_mat = sparse.load_npz(abs_path+"models/co_occurence.npz")

# Create svd matrix and take top n features, here 350 is used
svd = TruncatedSVD(n_components=350)
svd_mat = svd.fit_transform(co_occ_mat)

print("Explained Variance Ratio: ", svd.explained_variance_ratio_)
print("Eigen Values: ", svd.singular_values_)

print("The shape SVD matrix obtained using top k eigen-vectors: ", svd_mat.shape)

np.savez_compressed(abs_path+'models/svd_matrix.npz', a=svd_mat)
