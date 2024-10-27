import pickle
import socket
import struct
import cv2
import numpy as np
import matplotlib.pyplot as plt
import hashlib
import ocrspace


api = ocrspace.API()


# Assuming the database of students is as follows
database_info = {
    "23BTBOA13": "PRANAV SHAILESH GHANTE",
    "21CSBOA63": "UTKARSH RAJ",
    "21CSBOB53": "SHUBHAM SINGHA ROY",
    "23CSBOA18": "RITIK RAJ YADAV",
}

database_face = {
    "23BTBOA13": "1554e9c2363cdcd9bed291916302234cf0b0bfbdc5a21dcc021bc89684131a44",
    "23CSBOB53": "115f020efc24a11318943259cf7721bb66c319c3e2c9d03b30c39142f76eebf7",
    "23CSBOA18": "20cdf48263d24edb7732f05fc9e1ed3565c0b7fe706613af7711ea389b9cae6f",
}


# RSA

# STEP 1: Generate Two Large Prime Numbers (p,q) randomly
from random import randrange, getrandbits


def power(a, d, n):
    ans = 1
    while d != 0:
        if d % 2 == 1:
            ans = ((ans % n) * (a % n)) % n
        a = ((a % n) * (a % n)) % n
        d >>= 1
    return ans


def MillerRabin(N, d):
    a = randrange(2, N - 1)
    x = power(a, d, N)
    if x == 1 or x == N - 1:
        return True
    else:
        while d != N - 1:
            x = ((x % N) * (x % N)) % N
            if x == 1:
                return False
            if x == N - 1:
                return True
            d <<= 1
    return False


def is_prime(N, K):
    if N == 3 or N == 2:
        return True
    if N <= 1 or N % 2 == 0:
        return False

    # Find d such that d*(2^r)=X-1
    d = N - 1
    while d % 2 != 0:
        d /= 2

    for _ in range(K):
        if not MillerRabin(N, d):
            return False
    return True


def generate_prime_candidate(length):
    # generate random bits
    p = getrandbits(length)
    # apply a mask to set MSB and LSB to 1
    # Set MSB to 1 to make sure we have a Number of 1024 bits.
    # Set LSB to 1 to make sure we get a Odd Number.
    p |= (1 << length - 1) | 1
    return p


def generatePrimeNumber(length):
    A = 4
    while not is_prime(A, 128):
        A = generate_prime_candidate(length)
    return A


length = 5
# give me a code to add two numbers
P = 23
Q = 29
# print(P)
# print(Q)


# Step 2: Calculate N=P*Q and Euler Totient Function = (P-1)*(Q-1)
N = P * Q
eulerTotient = (P - 1) * (Q - 1)
# print(N)
# print(eulerTotient)


# Step 3: Find E


def GCD(a, b):
    if a == 0:
        return b
    return GCD(b % a, a)


E = generatePrimeNumber(4)
while GCD(E, eulerTotient) != 1:
    E = generatePrimeNumber(4)
# print(E)


# Step 4: Find D.


def gcdExtended(E, eulerTotient):
    a1, a2, b1, b2, d1, d2 = 1, 0, 0, 1, eulerTotient, E

    while d2 != 1:

        # k
        k = d1 // d2

        # a
        temp = a2
        a2 = a1 - (a2 * k)
        a1 = temp

        # b
        temp = b2
        b2 = b1 - (b2 * k)
        b1 = temp

        # d
        temp = d2
        d2 = d1 - (d2 * k)
        d1 = temp

        D = b2

    if D > eulerTotient:
        D = D % eulerTotient
    elif D < 0:
        D = D + eulerTotient

    return D


D = gcdExtended(E, eulerTotient)
# print(D)

print("The public of Authority is (", E, " , ", N, " )")
print("The private of Authority is (", D, " , ", N, " )")

# ------------------------------------------------------------------------------#

# Establish a socket connection
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ("127.0.0.1", 1234)
server.bind(server_address)
server.listen()
print("Server started listening")

# Accept a connection from the client
client, client_address = server.accept()
print("Connected to client")

# Sending Public key to the client
client.send(str(E).encode())
print(client.recv(1024).decode())
client.send(str(N).encode())


# Receive the size and data of the encryption matrix
enc_size_data = client.recv(4)
enc_size = struct.unpack("=L", enc_size_data)[0]
enc_data = b""
while len(enc_data) < enc_size:
    enc_data += client.recv(4096)
enc_matrix = pickle.loads(enc_data)

# Acknowledgement
client.send("Data Received Successfully".encode())

# Receive the size and data of the original image
img_size_data = client.recv(4)
img_size = struct.unpack("=L", img_size_data)[0]
img_data = b""
while len(img_data) < img_size:
    img_data += client.recv(4096)
original_img = pickle.loads(img_data)


cv2.imshow("Received Image", original_img)
cv2.waitKey(0)

print("Decrypting....")
# --------------------------------------------------------------------------#
row, col = original_img.shape[0], original_img.shape[1]
# Step 6: Decryption
for i in range(row):
    for j in range(col):
        r, g, b = enc_matrix[i][j]
        M1 = power(r, D, N)
        M2 = power(g, D, N)
        M3 = power(b, D, N)
        original_img[i, j] = [M1 % 256, M2 % 256, M3 % 256]

# Display the decrypted image
plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
plt.title("Decrypted Image")
plt.show()

# Save the image
cv2.imwrite("input.png", original_img)

# ---------------------------------------------------------------#
# Document Validation
print("Waiting for the Document to get validated..")
print("Extracting text from image using OCR:")
info = api.ocr_file("input.png")
words = info.split()

key = words[22]
value = words[15]

flag = 1

# verifying face in the image
face = [[0 for x in range(400)] for y in range(400)]

for i in range(211, 561):
    for j in range(130, 486):
        face[i - 211][j - 130] = original_img[i][j]

bytes_face = pickle.dumps(face)

# Compute the SHA-256 hash
hash_object = hashlib.sha256(bytes_face)
hex_dig = hash_object.hexdigest()

# print("SHA-256 hash of the list 'face':", hex_dig)


print(key, ",", value)
if key in database_info:
    print(database_info[key])

    if value in database_info[key] and hex_dig in database_face[key]:
        print("Document Submited is Valid")
    else:
        print("Tampered Document !! ")
        flag = 0
else:
    print("Tampered Document !! ")
    flag = 0

if flag == 1:
    client.send("Valid".encode())
    # --------------------------------------------------------------#
    my_tick = cv2.imread("tick.jpg")

    row, col = my_tick.shape[0], my_tick.shape[1]

    green = []
    tick_r = 72
    tick_g = 173
    tick_b = 65

    for i in range(row):
        for j in range(col):
            r, g, b = my_tick[i][j]
            if g == 173:
                green.append([i, j])

    for i in range(len(green)):
        val = green[i]
        original_img[val[0] + 300][val[1] + 400] = [tick_r, tick_g, tick_b]

    # Display the modified image
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title("Verified Image with sign")
    plt.show()

    # ------------------------------------------------------------------------#
    # Computing the hash of the document
    convert_bytes = bytes(original_img)
    msg_hash = hashlib.sha256(convert_bytes)
    msg = msg_hash.hexdigest()
    # print("The hash value of the image generated is ", msg)

    # Sending the certified Document
    img_data = pickle.dumps(original_img)
    img_size = struct.pack("=L", len(img_data))
    client.sendall(img_size)
    client.sendall(img_data)

    # Acknowledgement
    print(client.recv(1024).decode())

    # Sending the encrypted hash value
    # Encrypting using Rsa

    encrypt_msg = ""
    for i in range(len(msg)):
        val = ord(msg[i])
        encrypt_msg += chr(power(val, D, N))

    print("The Encrypted Msg is ", encrypt_msg)

    client.send(encrypt_msg.encode())
else:
    client.send("InValid".encode())
# Close the connection
client.close()
server.close()
