from typing import List


def closest_pair(arrays: List[int]):
    res = abs(arrays[0] - arrays[1])
    arrays = sorted(arrays)
    for i, value in enumerate(arrays):
        if i + 1 < len(arrays):
            next_value = arrays[i + 1]
            res = min(res, abs(next_value - value))
            if res == 0:
                return res
    return res


if __name__ == "__main__":
    print(closest_pair([-10, 3, 7, 4, 20]))
