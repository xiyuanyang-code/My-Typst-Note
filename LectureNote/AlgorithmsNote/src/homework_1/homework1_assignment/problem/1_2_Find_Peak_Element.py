"""
Problem 1-2: Find Peak Element
EN: Return the index of any peak element (strictly greater than neighbors) in O(log n) time.
CN: 在 O(log n) 时间内返回任意一个峰值元素的下标（严格大于相邻元素）。
"""

class Solution(object):
    def findPeakElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        length = len(nums)
        if length == 0:
            return -1
        if length == 1:
            return 0
        
        left = 0
        right = length - 1 
        
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] < nums[mid + 1]:
                left = mid + 1
            elif nums[mid] > nums[mid + 1]:
                right = mid

        return left
        
