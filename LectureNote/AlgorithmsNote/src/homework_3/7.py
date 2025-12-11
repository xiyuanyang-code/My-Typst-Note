from typing import List


def count_paths(F: List[List[int]]):
    n = len(F)

    # K[i][j] 代表最大蘑菇数量
    # X[i][j] 代表拾取最大蘑菇数量的路径数量

    K = [[-float("inf")] * n for _ in range(n)]
    X = [[0] * n for _ in range(n)]

    # ensure [0][0] and [n-1][n-1] does not have trees

    for i in range(n):
        for j in range(n):
            if F[i][j] == "t":
                K[i][j] = -float("inf")
                X[i][j] = 0
                continue
            if i == 0 and j == 0:
                K[0][0] = 1 if F[0][0] == "m" else 0
                X[0][0] = 1
                continue

            # compute i,j
            candidates = []
            if i - 1 >= 0 and K[i - 1][j] > -float("inf"):
                candidates.append((K[i - 1][j], X[i - 1][j]))
            if j - 1 >= 0 and K[i][j - 1] >-float("inf"):
                candidates.append((K[i][j - 1], X[i][j - 1]))

            if not candidates:
                K[i][j] = -float("inf")
                X[i][j] = 0
                continue

            max_mushrooms = max(c[0] for c in candidates)
            total_x = sum(c[1] for c in candidates if c[0] == max_mushrooms)

            K[i][j] = max_mushrooms
            X[i][j] = total_x

            if F[i][j] == "m":
                K[i][j] += 1

    return X[n - 1][n - 1]


if __name__ == "__main__":
    n = int(input())

    raw_input_lines = []
    F = []

    for _ in range(n):
        line = input().strip()
        raw_input_lines.append(line)
        F.append(list(line))

    for i in range(n):
        for j in range(n):
            if F[i][j] not in ("m", "t", "x"):
                print("Error")

    paths = count_paths(F)
    print(paths)

    # if paths == 1430291113574400:
    #     print(raw_input_lines[0])
    #     print(F[0])
    # else:
    #     print(paths)
