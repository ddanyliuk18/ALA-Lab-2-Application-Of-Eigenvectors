# ALA-Lab-2-Application-Of-Eigenvectors
 
# Task 1: Computing Eigenvalues and Eigenvectors

## Description
This function computes eigenvalues and eigenvectors of a square matrix using NumPy and verifies the fundamental eigenvalue equation: **A·v = λ·v**

## Code
```python
import numpy as np

def find_eigen(square_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(square_matrix)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    return eigenvalues, eigenvectors

test_matrix = np.array([[2, 3, 4], [1, 9, 4], [4, 3, 2]])
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
```

## How it works
1. **find_eigen()** - extracts eigenvalues and eigenvectors using `np.linalg.eig()` and converts them to real numbers
2. **Verification loop** - for each eigenvalue λ and eigenvector v, checks if A·v equals λ·v using `np.isclose()` for numerical precision

## Output
```
Checking for λ0..
A*v = [-1.19904022 -0.44964008  1.53627028]
λ*v = [-1.19904022 -0.44964008  1.53627028]
True
Checking for λ1..
A*v = [-2.00640578  1.77454014 -2.00640578]
λ*v = [-2.00640578  1.77454014 -2.00640578]
True
Checking for λ2..
A*v = [-4.94606353 -9.32054666 -4.94606353]
λ*v = [-4.94606353 -9.32054666 -4.94606353]
True
```

All three eigenvalue-eigenvector pairs satisfy the equation A·v = λ·v

---

# Task 2: Image Compression Using PCA

## Description
Implementation of image dimensionality reduction using Principal Component Analysis (PCA). The program compresses a color image by converting it to grayscale and reconstructing it with varying numbers of principal components.

## Code
```python
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def pca_restore(image, k):
    if k < 1:
        pca = PCA(n_components=k)
        pca.fit(image)
        compressed = pca.transform(image)
        image_restored = pca.inverse_transform(compressed)
        variance_ratio = pca.explained_variance_ratio_
        print("Variance Ratio for each component:")
        print(variance_ratio)
        cumulative_variance = np.cumsum(variance_ratio)
        print("Cumulative Variance:")
        print(cumulative_variance)
        components = pca.n_components_
        return image_restored, components
    else:
        pca = PCA(n_components=int(k))
        pca.fit(image)
        compressed = pca.transform(image)
        image_restored = pca.inverse_transform(compressed)
        return image_restored, int(k)

def full_pca(image):
    pca_full = PCA()
    pca_full.fit(image)
    explained_variance = pca_full.explained_variance_ratio_
    cumulative = np.cumsum(explained_variance)
    return cumulative

def graph(components, cumulative, achieved):
    plt.figure()
    plt.plot(components, cumulative)
    plt.xlabel("Principal components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Cumulative Explained Variance explained by the components")
    plt.axhline(y=0.95, color='r', linestyle='--')
    plt.axvline(x=achieved, color='k', linestyle='--')

image_raw = imread("images/ala_photo.jpg")
print(image_raw.shape)
image_sum = image_raw.sum(axis=2)
print(image_sum.shape)
image_bw = image_sum/image_sum.max()
print(image_bw.max())
image_restored, components = pca_restore(image_bw, 0.95)
image_restored1, comp1 = pca_restore(image_bw, 329)
image_restored2, comp2 = pca_restore(image_bw, 81)
image_restored3, comp3 = pca_restore(image_bw, 17)
cumulative = full_pca(image_bw)
comp_num = np.arange(1, len(cumulative) + 1)
graph(comp_num, cumulative, components)

fig, axes = plt.subplots(2, 3, figsize=(18, 8))
axes[0, 0].imshow(image_raw)
axes[0, 0].set_title("Original RGB")
axes[0, 0].axis('off')
axes[0, 1].imshow(image_bw, cmap='gray')
axes[0, 1].set_title("B&W")
axes[0, 1].axis('off')
axes[0, 2].imshow(image_restored, cmap='gray')
axes[0, 2].set_title("PCA\n95% coverage")
axes[0, 2].axis('off')
axes[1, 0].imshow(image_restored1, cmap='gray')
axes[1, 0].set_title("PCA\n329 components")
axes[1, 0].axis('off')
axes[1, 1].imshow(image_restored2, cmap='gray')
axes[1, 1].set_title("PCA\n81 components")
axes[1, 1].axis('off')
axes[1, 2].imshow(image_restored3, cmap='gray')
axes[1, 2].set_title("PCA\n17 components")
axes[1, 2].axis('off')
plt.tight_layout()
plt.show()
```

## How it works
## Applying PCA on photo
<img width="1280" height="628" alt="image" src="https://github.com/user-attachments/assets/295d064f-6623-4b9b-8c9e-1b4aae665f78" />

## Graph result
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/4c1616a5-5c48-4a27-b1c6-b81edce18c8a" />

### 1. Image Preprocessing
- Load RGB image and display its shape
- Convert to grayscale by summing RGB channels and normalizing

### 2. PCA Functions
- **pca_restore(image, k)** - reconstructs image using k components (or variance ratio if k < 1)
- **full_pca(image)** - computes cumulative variance for all components
- **graph()** - visualizes cumulative variance with 95% threshold

### 3. Image Reconstruction
- 95% variance coverage - captures main features while losing fine details
- 329 components - nearly full detail restoration
- 81 components - good balance between compression and quality
- 17 components - aggressive compression, basic structure only

## Results
The program generates:
1. **Cumulative variance graph** showing how many components achieve 95% coverage
2. **Comparison grid** with 6 images:
   - Original RGB
   - Grayscale conversion
   - PCA reconstructions with different component counts

### Observations
- **More components** → sharper image, less compression
- **Fewer components** → blurrier image, more compression
- **95% variance** preserves object recognition while sacrificing sharpness

---

# Task 3: Cryptography Using Matrix Diagonalization

## Description
Implementation of encryption and decryption functions using matrix diagonalization with eigenvalues and eigenvectors. The program converts text messages into numerical vectors, encrypts them using a key matrix, and decrypts them back to the original text.

## Code
```python
import numpy as np

def encrypt_message(message, key_matrix):
    message_vector = np.array([ord(char) for char in message])
    eigenvalues, eigenvectors = np.linalg.eig(key_matrix)
    diagonalized_key_matrix = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)), np.linalg.inv(eigenvectors))
    encrypted_vector = np.dot(diagonalized_key_matrix, message_vector)
    return encrypted_vector

def decrypt_message(encrypted_vector, key_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(key_matrix)
    decrypt_vector = np.dot(np.dot(np.dot(eigenvectors, np.diag(1 / eigenvalues)), np.linalg.inv(eigenvectors)), encrypted_vector)
    chars = []
    for i in decrypt_vector:
        ascii_rep = int(round(i.real))
        chars.append(chr(ascii_rep))
    message = ''.join(chars)
    return message

message = "The more knowledge I gain, the clearer it becomes that there's so much I'm unaware of"
key_matrix = np.random.randint(0, 256, (len(message), len(message)))
encrypted = encrypt_message(message, key_matrix)
print(f"Encrypted string: {encrypted}")
decrypted = decrypt_message(encrypted, key_matrix)
print(f"Decrypted string: {decrypted}")
```

## How it works

### Encryption Process
1. Convert message characters to ASCII values (message vector)
2. Compute eigenvalues and eigenvectors of the key matrix
3. Reconstruct the diagonalized key matrix: **A = PDP⁻¹**
4. Multiply the diagonalized matrix by the message vector to encrypt

### Decryption Process
1. Compute eigenvalues and eigenvectors of the same key matrix
2. Use inverse diagonalization: multiply by **P(D⁻¹)P⁻¹**
3. Convert resulting numerical vector back to ASCII characters
4. Join characters to recover the original message

### Key Components
- **Key Matrix**: Randomly generated square matrix (size = message length)
- **Diagonalization**: Uses eigendecomposition for encryption/decryption
- **ASCII Conversion**: Maps characters ↔ numbers for mathematical operations

## Output
```
Encrypted string: [2564692. -1.36340055e-08j 2384600.99999999-6.69028021e-09j
 1741446. -2.03434378e-09j ... 2001949.99999999-4.00373368e-09j]
Decrypted string: The more knowledge I gain, the clearer it becomes that there's so much I'm unaware of
```

The decrypted message perfectly matches the original, confirming the encryption/decryption system works correctly using matrix diagonalization.

## Mathematical Foundation
The security relies on the difficulty of computing eigenvalues and eigenvectors without knowing the key matrix. The small imaginary components in encrypted values are numerical artifacts from floating-point calculations.

