import numpy as np

def test_1():
    M = 100
    N = 100
    X = np.random.rand(M,N)
    row_means = np.mean(X, axis=1)
    GT = X - row_means[:, np.newaxis]

    ANS_1 = X - np.mean(X, axis=1, keepdims=True)
    ANS_2 = X - np.expand_dims(np.mean(X, axis=1), 1)

    w = np.random.rand(N,)
    print(X)
    print(GT)
    print(GT == ANS_1)
    print(GT == ANS_2)

    return (X)


if __name__ == "__main__":
    test_1()