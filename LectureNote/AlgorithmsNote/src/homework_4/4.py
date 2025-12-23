from typing import List


def get_medium(nums1: List[int], nums2: List[int]):
    length_1 = len(nums1)
    length_2 = len(nums2)

    if (length_1 + length_2) % 2 == 0:
        return (
            find_k_element(nums1, nums2, (length_1 + length_2) / 2)
            + find_k_element(nums1, nums2, (length_1 + length_2) / 2 + 1)
        ) / 2
    else:
        return find_k_element(nums1, nums2, (length_1 + length_2) // 2)

def find_k_element(nums1: List[int], nums2: List[int], k: int) -> int:
    def get_kth(i1: int, i2: int, k: int) -> int:
        if i1 == len(nums1):
            return nums2[i2 + k - 1]
        if i2 == len(nums2):
            return nums1[i1 + k - 1]
        if k == 1:
            return min(nums1[i1], nums2[i2])

        half = k // 2
        idx1 = min(i1 + half, len(nums1)) - 1
        idx2 = min(i2 + half, len(nums2)) - 1
        
        pivot1, pivot2 = nums1[idx1], nums2[idx2]

        if pivot1 <= pivot2:
            return get_kth(idx1 + 1, i2, k - (idx1 - i1 + 1))
        else:
            return get_kth(i1, idx2 + 1, k - (idx2 - i2 + 1))

    return get_kth(0, 0, k)

if __name__ == "__main__":
    input_str_1 = input()
    input_str_2 = input()
    input_list_1 = [int(x.strip()) for x in input_str_1.split(",")]
    input_list_2 = [int(x.strip()) for x in input_str_2.split(",")]
    print(get_medium(input_list_1, input_list_2))