from typing import List, Tuple


def closest_pair(arrays: List[Tuple[int, int]]):
    if len(arrays) == 1:
        return None
    if len(arrays) == 2:
        return (arrays[0][0] - arrays[1][0]) ** 2 + (arrays[0][1] - arrays[1][1]) ** 2
    if len(arrays) == 3:
        return min(
            (arrays[0][0] - arrays[1][0]) ** 2 + (arrays[0][1] - arrays[1][1]) ** 2,
            (arrays[1][0] - arrays[2][0]) ** 2 + (arrays[1][1] - arrays[2][1]) ** 2,
            (arrays[0][0] - arrays[2][0]) ** 2 + (arrays[0][1] - arrays[2][1]) ** 2,
        )

    sorted_points = sorted(arrays, key=lambda x: x[0])
    # do divide and conquer
    pivot = len(sorted_points) // 2
    left_closest_res = closest_pair(arrays=arrays[:pivot])
    right_closest_res = closest_pair(arrays=arrays[pivot:])
    res = min(left_closest_res, right_closest_res)

    strip = []
    strip.append(arrays[pivot])
    pivot_x = arrays[pivot][0]
    for i in range(pivot - 1, -1, -1):
        if abs(arrays[i][0] - pivot_x) <= res:
            strip.append(arrays[i])
        else:
            break
    for j in range(pivot + 1, len(arrays)):
        if abs(arrays[j][0] - pivot_x) <= res:
            strip.append(arrays[i])
        else:
            break

    sorted_strip = sorted(strip, key=lambda x: x[1])
    for i, candidate in enumerate(sorted_strip):
        for j in range(i + 1, len(sorted_strip)):
            if abs(arrays[i][1] - arrays[j][1]) >= res:
                continue
            res = min(
                res,
                (arrays[i][0] - arrays[j][0]) ** 2 + (arrays[i][1] - arrays[j][1]) ** 2,
            )
    return res


if __name__ == "__main__":
    print(closest_pair([(1, 2), (2, 3), (4, 5), (1, 8)]))
