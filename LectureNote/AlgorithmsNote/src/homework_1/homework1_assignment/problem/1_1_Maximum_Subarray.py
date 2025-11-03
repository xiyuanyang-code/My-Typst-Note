"""
Problem 1-1: Maximum Subarray
EN: Given an integer array, find the contiguous subarray with the largest sum and return that sum.
CN: 给定整数数组，找到和最大的连续子数组，并返回其和。
"""

class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        length = len(nums)
        if length == 0:
            return 0
        if length == 1:
            return nums[0]
        elif length == 2:
            return max(nums[0], nums[1], nums[0] + nums[1])
        split = length // 2
        # 0..split
        # split..length
        sum_left = self.maxSubArray(nums=nums[:split])
        sum_right = self.maxSubArray(nums=nums[split:])
        # compute medium max sum
        sum_medium_left = 0
        sum_medium_max_left = nums[split - 1]
        sum_medium_right = 0
        sum_medium_max_right = nums[split]
        for i in range(split - 1, -1, -1):
            sum_medium_left += nums[i]
            sum_medium_max_left = max(sum_medium_max_left, sum_medium_left)
        for j in range(split, length):
            sum_medium_right += nums[j]
            sum_medium_max_right = max(sum_medium_max_right, sum_medium_right)

        return max(sum_medium_max_left + sum_medium_max_right, sum_left, sum_right)
        
