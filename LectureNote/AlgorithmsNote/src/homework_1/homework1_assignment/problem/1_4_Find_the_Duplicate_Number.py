"""
Problem 1-4: Find the Duplicate Number
EN: Given n+1 integers in [1..n], find the single duplicate without modifying the array and using only O(1) extra space.
CN: 给定位于区间 [1..n] 的 n+1 个整数，在不修改数组且仅用 O(1) 额外空间的前提下找出唯一的重复数。
"""

class Solution(object):
    def findDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # using floyd circle algorithms
        slow = 0
        fast = 0
        while True:
            slow = nums[slow]
            fast = nums[fast]
            fast = nums[fast]
            if slow == fast:
                break
        
        new_ptr = 0
        while new_ptr != slow:
            slow = nums[slow]
            new_ptr = nums[new_ptr]
        return slow


        

        
