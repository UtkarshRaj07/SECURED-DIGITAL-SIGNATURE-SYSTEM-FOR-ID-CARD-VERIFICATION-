import socket
import cv2
import struct
import pickle
import matplotlib.pyplot as plt
import hashlib


# RSA

# STEP 1: Generate Two Large Prime Numbers (p,q) randomly
# Load the image from a local file path
my_img = cv2.imread("pranav.jpeg")

# Display the original image
plt.imshow(cv2.cvtColor(my_img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.show()


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
P = 17
Q = 37

# print(P)
# print(Q)


# Step 2: Calculate N=P*Q and Euler Totient Function = (P-1)*(Q-1)
N = P * Q
eulerTotient = (P - 1) * (Q - 1)
print(N)
print(eulerTotient)


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

print("The public of Client is (", E, " , ", N, " )")
print("The private of Client is (", D, " , ", N, " )")

# -------------------------------------------------------------------------------#
conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ("192.168.201.217", 1234)
conn.connect(server_address)

# Receiving Public key of server
E_server = int(conn.recv(1024).decode())
conn.send("Acknowledgement".encode())
N_server = int(conn.recv(1024).decode())
print("The Public key of Authority is (", E_server, ",", N_server)


# --------------------------------------------------------------------------------#

row, col = my_img.shape[0], my_img.shape[1]
enc = [[0 for x in range(3000)] for y in range(3000)]

# Step 5: Encryption
for i in range(row):
    for j in range(col):
        r, g, b = my_img[i, j]
        C1 = power(r, E_server, N_server)
        C2 = power(g, E_server, N_server)
        C3 = power(b, E_server, N_server)
        enc[i][j] = [C1, C2, C3]
        my_img[i, j] = [C1 % 256, C2 % 256, C3 % 256]


# ---------------------------------------------------------------------------------#

# Serialize the 'enc' list using pickle
enc_data = pickle.dumps(enc)
img_data = pickle.dumps(my_img)


# Serialize and send the encryption matrix
enc_size = struct.pack("=L", len(enc_data))
conn.sendall(enc_size)
conn.sendall(enc_data)
print("First data sent successfully")

# Waiting for Acknowledgment'
print(conn.recv(1024).decode())

# Serialize and send the original image
img_data = pickle.dumps(my_img)
img_size = struct.pack("=L", len(img_data))
conn.sendall(img_size)
conn.sendall(img_data)

# Document verification status
status = conn.recv(1024).decode()

print("Waiting for the document to get certified")
if status == "InValid":
    print("Document cannot be Certified")
else:
    # # Waiting for certified Document
    img_size_data = conn.recv(4)
    img_size = struct.unpack("=L", img_size_data)[0]
    img_data = b""
    while len(img_data) < img_size:
        img_data += conn.recv(4096)
    original_img = pickle.loads(img_data)

    print("Certified Documnet Received")

    cv2.imshow("Certified Document", original_img)
    cv2.waitKey(0)

    # Sending Acknowledgement
    conn.send("Received Successfully".encode())

    # The encrypted hash value
    hash_val = conn.recv(1024).decode()
    print("The Encrypted Hash Certificate ", hash_val)

    decrypt_msg = ""
    for i in range(len(hash_val)):
        val = ord(hash_val[i])
        decrypt_msg += chr(power(val, E_server, N_server))

    print("Verifying the Digitally Cerified and Signed Document")
    # Computing the hash of the document
    convert_bytes = bytes(original_img)
    msg_hash = hashlib.sha256(convert_bytes)
    msg = msg_hash.hexdigest()

    print("The Decrypted Hash is ", decrypt_msg)
    print("The Computed Hash is ", msg)
    if msg == decrypt_msg:
        print("The Digitally Certified Document is Valid ")

# Close the connection
conn.close()
