import numpy as np

def find_eigen(square_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(square_matrix)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    return eigenvalues, eigenvectors

test_matrix = np.array ([[2, 3, 4], [1, 9, 4], [4, 3, 2]])
eignvales, eignvectrs = find_eigen(test_matrix)

for i in range(len(eignvales)):
    v = eignvectrs[:, i]
    lambd = eignvales[i]
    res1 = test_matrix @ v
    res2 = lambd * v
    is_value = np.isclose(res1, res2)
    print(f"Checking for λ{i}..")
    print(f"A*v = {res1}")
    print(f"λ*v = {res2}")
    if np.all(is_value):
        print("True")
    else:
        print("False")