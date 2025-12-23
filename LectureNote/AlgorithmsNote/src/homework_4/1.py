from typing import List

def count_reverse_pairs(nums: List[int]) -> int:
    if not nums:
        return 0
    
    def merge_sort(left, right):
        if left >= right:
            return 0
        
        mid = (left + right) // 2
        count = merge_sort(left, mid) + merge_sort(mid + 1, right)
        
        j = mid + 1
        for i in range(left, mid + 1):
            while j <= right and nums[i] > 2 * nums[j]:
                j += 1
            count += (j - (mid + 1))

        nums[left:right + 1] = sorted(nums[left:right + 1])
        return count

    return merge_sort(0, len(nums) - 1)



if __name__ == "__main__":
    raw_input = input()
    input_list = [int(x.strip()) for x in raw_input.split(",")]
    count = count_reverse_pairs(input_list)
    print(count)
