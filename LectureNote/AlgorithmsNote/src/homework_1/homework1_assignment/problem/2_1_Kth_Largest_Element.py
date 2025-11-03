"""
Problem 2-1: Kth Largest Element in an Array
EN: Return the k-th largest element in the array (not necessarily distinct) without fully sorting.
CN: 在不整体排序的情况下，返回数组中的第 k 大元素（不要求元素互异）。
"""

class Solution(object):
    def findKthsmallest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        import random
        if len(nums) < k:
            return None
        if len(nums) == 1:
            return nums[0]
        # select pivot
        pivot = nums[random.randrange(len(nums))]
        # split to two sub-arrays
        left_sub_array = []
        right_sub_array = []
        same_value_count = 0
        for num in nums:
            if num < pivot:
                left_sub_array.append(num)
            elif num > pivot:
                right_sub_array.append(num)
            else:
                same_value_count += 1
        if len(left_sub_array) < k and len(left_sub_array) + same_value_count >= k:
            return pivot
        elif len(left_sub_array) >= k:
            return self.findKthsmallest(left_sub_array, k)
        else:
            return self.findKthsmallest(right_sub_array,k-len(left_sub_array)-same_value_count)
    
    def findKthLargest(self, nums, k):
        return self.findKthsmallest(nums, len(nums) + 1 - k)
