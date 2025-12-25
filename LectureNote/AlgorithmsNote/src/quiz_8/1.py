from typing import List
import random


def quick_select(array: List[int], k: int):
    length = len(array)
    if length < k:
        return None
    if length == 1:
        return array[0]

    # select pivot randomly
    random_pivot = random.randint(0, length - 1)
    pivot_value = array[random_pivot]
    left_array = []
    right_array = []
    same_value_count = 0
    for value in array:
        if value < pivot_value:
            left_array.append(value)
        elif value > pivot_value:
            right_array.append(value)
        else:
            same_value_count += 1

    if len(left_array) >= k:
        return quick_select(left_array, k)
    elif len(left_array) < k and len(left_array) + same_value_count >= k:
        return pivot_value
    else:
        return quick_select(right_array, k - len(left_array) - same_value_count)


if __name__ == "__main__":
    for i in range(1,7):
        print(quick_select([7,1,5,2,2,9], i))