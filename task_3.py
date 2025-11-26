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
message = "The more knowledge I gain, the clearer it becomes that there’s so much I’m unaware of"
key_matrix = np.random.randint(0, 256, (len(message), len(message)))
encrypted = encrypt_message(message, key_matrix)
print (f"Encrypted string: {encrypted}")
decrypted = decrypt_message(encrypted, key_matrix)
print (f"Decrypted string: {decrypted}")