import numpy as np

def Jacobi(x):
    return np.array([(1 + x[1]) / 2, (8 + x[0] + x[2]) / 3, (-5 + x[1]) / 2])

def GS(x):
    y = np.copy(x)
    y[0] = np.copy((1 + y[1]) / 2)
    y[1] = (8 + y[0] + y[2]) / 3
    y[2] = (-5 + y[1]) / 2
    return np.round(y, decimals=4)


print("---------- Jacobi ---------")
x = np.zeros(3)
for i in range(1, 21):
    x = Jacobi(x)
    print(f"[{i}] {x}")
    x = Jacobi(x)

print("---------- G-S -----------")
x = np.zeros(3)
for i in range(1, 21):
    x = GS(x)
    print(f"[{i}] {x}")