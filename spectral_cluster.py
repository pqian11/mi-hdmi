from sklearn.cluster import SpectralClustering
import numpy as np
import pickle
import timeit
import sys

f = open('vocab_similarity_matrix.pkl', 'rb')
X = pickle.load(f)
f.close()

print(X.shape)

X = (X+1)/2

num_of_clusters = int(sys.argv[1])

print('Start spectral clustering...')

start = timeit.default_timer()

clustering = SpectralClustering(n_clusters=num_of_clusters, assign_labels="discretize", affinity='precomputed').fit_predict(X)
print(clustering[:100])

stop = timeit.default_timer()

print('Complete clustering in', start-stop)

outfile = open('vocab_clusters_'+str(num_of_clusters)+'.pkl', 'wb')
pickle.dump(clustering, outfile, protocol=4)
outfile.close()
