import numpy as np
X = np.array([
    [78, 85, 80, 82],
    [65, 70, 68, 72],
    [90, 92, 88, 91],
    [72, 75, 70, 74],
    [85, 88, 84, 86]
])
print("Original Dataset:\n", X)
mean = np.mean(X, axis=0)
X_centered = X - mean
print("\nMean of each attribute:\n", mean)
print("\nMean Centered Data:\n", X_centered)
cov_matrix = np.cov(X_centered.T)
print("\nCovariance Matrix:\n", cov_matrix)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print("\nEigenvalues:\n", eigenvalues)
print("\nEigenvectors:\n", eigenvectors)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]
print("\nSorted Eigenvalues:\n", eigenvalues)
print("\nSorted Eigenvectors:\n", eigenvectors)
pc1 = eigenvectors[:, 0]
print("\nFirst Principal Component (PC1 Eigenvector):\n", pc1)
pc1_scores = np.dot(X_centered, pc1)
print("\nPC1 Scores for each student:\n", pc1_scores)
