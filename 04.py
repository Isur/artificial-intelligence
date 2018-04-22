from time import time
from numpy.random import RandomState
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.cluster import MiniBatchKMeans
from sklearn import decomposition
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSCanonical
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import math

# # Zadanie 1:

# n_row, n_col = 2, 3
# n_components = 1 
# image_shape = (64, 64)
# rng = RandomState(0)

# dataset = fetch_olivetti_faces(shuffle=True, random_state=rng)
# faces = dataset.data

# n_samples, n_features = faces.shape

# faces_centered = faces - faces.mean(axis=0)
# faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)
# print("Dataset consists of %d_faces" %n_samples)

# def plot_gallery(title,images,n_col=n_col, n_row=n_row):
#     plt.figure(figsize=(2*n_col,2.26*n_row))
#     plt.suptitle(title,size=16)
#     for i, comp in enumerate(images):
#         plt.subplot(n_row,n_col,i+1)
#         vmax = max(comp.max(), -comp.min())
#         plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray, interpolation='nearest',vmin=-vmax,vmax=vmax)
#         plt.xticks(())
#         plt.yticks(())
#     plt.subplots_adjust(0.01,0.05,0.99,0.93,0.04,0.)

# estimators = [
#         ('Eigenfaces - PCA using randomized SVD',
#          decomposition.PCA(n_components=n_components,
#          svd_solver = 'randomized', whiten=True),True)
#         ]
# plot_gallery("First centered Olivetti_faces", faces_centered[:n_components])

# for name, estimator, center in estimators:
#     print("Extracting the top \%d \%s..." %(n_components,name))
#     t0 = time()
#     data = faces
#     if center:
#         data =faces_centered
#         estimator.fit(data)
#         print(estimator.score(data))
#         train_time = (time() - t0)
#         print("Done in %0.3fs" %train_time)
#         if hasattr(estimator, 'cluster_centers_'):
#             compoonents_ = estimator.cluster_centers_
#         else:
#             components_ = estimator.components_
            
#         if (hasattr(estimator, 'noise_variance_') and estimator.noise_variance_.ndim > 0):
#             plot_gallery("Pixelwise variance", estimator.noise_variance_.reshape(1,-1), n_col=1,n_row=1)
#         plot_gallery('%s - Train time %.1fs'  %(name,train_time), components_[:n_components])
#     plt.show()

# Zadanie 2:

mnist = fetch_mldata('MNIST original')
train, test, train_targets, test_targets = train_test_split(mnist.data, mnist.target, test_size=0.50, random_state = 42)

# Zadanie 3:
# max_value = 0
# max_number = 0
# for i in range(1,6):
#     lda = LDA(n_components=i)
#     lda_train= lda.fit(train, train_targets).transform(train)
#     lda_test= lda.fit(test, test_targets).transform(test)
#     knn = KNeighborsClassifier(round(math.sqrt(mnist.data.shape[0])), metric='euclidean', weights='uniform')
#     knn.fit(lda_train,train_targets)
#     print("Score for ", i, " components: ", knn.score(lda_test,test_targets))
#     if max_value < knn.score(lda_test,test_targets):
#         max_value = knn.score(lda_test,test_targets)
#         max_number = i
# print("Max for: ", max_number, " is: ", max_value)

# Zadanie 4:
# max_value = 0
# max_number = 0
# for i in range(1,6):
#     plsca = PLSCanonical(n_components=i)
#     plsca.fit(train,train_targets)
#     pls_train= plsca.fit(train, train_targets).transform(train)
#     pls_test= plsca.fit(test, test_targets).transform(test)
#     knn = KNeighborsClassifier(round(math.sqrt(mnist.data.shape[0])), metric='euclidean', weights='uniform')
#     knn.fit(pls_train,train_targets)
#     print("Score for ", i, " components: ", knn.score(pls_test,test_targets))
#     if max_value < knn.score(pls_test,test_targets):
#         max_value = knn.score(pls_test,test_targets)
#         max_number = i
# print("Max for: ", max_number, " is: ", max_value)



# Zadanie 5:
knn = KNeighborsClassifier(round(math.sqrt(mnist.data.shape[0])))
sfs = SFS(knn, k_features=3,forward=True, floating=False, verbose=2,scoring='accuracy', cv=0)
