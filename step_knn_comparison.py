import numpy as np
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.metrics import accuracy_score
from time import time
from step1_data import trainloader, testloader

def dataloader_to_numpy(loader, max_samples=None):
    X_list, y_list = [], []
    seen = 0
    for imgs, labels in loader:
        b = imgs.shape[0]
        if max_samples is not None and seen + b > max_samples:
            cut = max_samples - seen
            imgs = imgs[:cut]
            labels = labels[:cut]
            b = cut
        X_list.append(imgs.view(b, -1).cpu().numpy().astype(np.float32))
        y_list.append(labels.cpu().numpy().astype(np.int64))
        seen += b
        if max_samples is not None and seen >= max_samples:
            break
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y

MAX_TRAIN = 20000   
MAX_TEST  = 5000    

print("Μετατροπή PyTorch DataLoader σε NumPy:")
X_train, y_train = dataloader_to_numpy(trainloader, MAX_TRAIN)
X_test,  y_test  = dataloader_to_numpy(testloader,  MAX_TEST)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# --------- Classifier 1: KNN (k=1) ---------
print("\n[KNN k=1]")
t0 = time()
knn1 = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
knn1.fit(X_train, y_train)
pred1 = knn1.predict(X_test)
acc1 = accuracy_score(y_test, pred1)
print(f"Accuracy: {acc1*100:.2f}%   (time: {time()-t0:.1f}s)")

# --------- Classifier 2: KNN (k=3) ---------
print("\n[KNN k=3]")
t0 = time()
knn3 = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
knn3.fit(X_train, y_train)
pred3 = knn3.predict(X_test)
acc3 = accuracy_score(y_test, pred3)
print(f"Accuracy: {acc3*100:.2f}%   (time: {time()-t0:.1f}s)")

# --------- Classifier 3: Nearest Centroid ---------
print("\n[Nearest Centroid]")
t0 = time()
nc = NearestCentroid()  # απόσταση ευκλείδεια γύρω από το μέσο κάθε κλάσης
nc.fit(X_train, y_train)
predc = nc.predict(X_test)
accc = accuracy_score(y_test, predc)
print(f"Accuracy: {accc*100:.2f}%   (time: {time()-t0:.1f}s)")

print("\n--- Summary ---")
print(f"KNN-1: {acc1*100:.2f}% | KNN-3: {acc3*100:.2f}% | Nearest-Centroid: {accc*100:.2f}%")
