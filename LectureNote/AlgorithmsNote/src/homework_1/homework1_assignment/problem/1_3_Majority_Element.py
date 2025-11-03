"""
Problem 1-3: Majority Element
EN: Return the element that appears more than floor(n/2) times (guaranteed to exist).
CN: 返回在数组中出现次数超过 ⌊n/2⌋ 的元素（保证存在）。
"""


class Solution(object):
    def count_occurrences(self, arr, start, end, target):
        """
        统计元素在指定范围内的出现次数

        参数：
        - arr: 数组
        - start: 起始索引
        - end: 结束索引
        - target: 目标元素

        返回：
        - target在arr[start:end+1]中的出现次数
        """
        count = 0
        for i in range(start, end + 1):
            if arr[i] == target:
                count += 1
        return count

    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if nums == []:
            return None
        return self.majority_helper(nums, 0, len(nums) - 1)

    def majority_helper(self, arr, left, right):
        """
        递归查找多数元素的辅助函数

        参数：
        - arr: 数组
        - left: 左边界
        - right: 右边界

        返回：
        - 在arr[left:right+1]范围内的多数元素候选
        """
        if left == right:
            return arr[left]

        mid = (left + right) // 2
        left_candidate = self.majority_helper(arr, left, mid)
        right_candidate = self.majority_helper(arr, mid+1, right)

        # validate left candidate
        if (
            self.count_occurrences(arr, left, right, target=left_candidate)
            >= (right - left + 1) // 2
        ):
            return left_candidate
        elif (
            self.count_occurrences(arr, left, right, target=right_candidate)
            >= (right - left + 1) // 2
        ):
            return right_candidate
        else:
            return None
