from typing import List


class Solution:
    def reversePairs(self, nums: List[int]) -> int:
        if not nums:
            return 0
        return self.merge_sort(nums, 0, len(nums) - 1)

    def merge_sort(self, nums, left, right):
        if left >= right:
            return 0

        mid = (left + right) // 2
        # 1. 递归计算左半边的逆序对
        # 2. 递归计算右半边的逆序对
        count = self.merge_sort(nums, left, mid) + self.merge_sort(nums, mid + 1, right)

        # 3. 计算跨越左右的逆序对，并进行归并排序
        temp = []
        i, j = left, mid + 1

        while i <= mid and j <= right:
            if nums[i] <= nums[j]:
                temp.append(nums[i])
                i += 1
            else:
                # 发现逆序对：nums[i] > nums[j]
                # 此时从 nums[i] 到 nums[mid] 的所有元素都大于 nums[j]
                count += mid - i + 1
                temp.append(nums[j])
                j += 1

        # 处理剩余元素
        while i <= mid:
            temp.append(nums[i])
            i += 1
        while j <= right:
            temp.append(nums[j])
            j += 1

        # 写回原数组
        nums[left : right + 1] = temp
        return count


from typing import List


class Solution:
    def countSmaller(self, nums: List[int]) -> List[int]:
        n = len(nums)
        self.res = [0] * n
        # 将数值和它的原始索引绑定，形如 [(v1, 0), (v2, 1), ...]
        # 这样排序后我们依然知道某个数原来的位置在哪里
        indexed_nums = list(enumerate(nums))
        self.merge_sort(indexed_nums)
        return self.res

    def merge_sort(self, arr):
        if len(arr) <= 1:
            return arr

        mid = len(arr) // 2
        left = self.merge_sort(arr[:mid])
        right = self.merge_sort(arr[mid:])

        return self.merge(left, right)

    def merge(self, left, right):
        merged = []
        i = 0  # 左侧数组指针
        j = 0  # 右侧数组指针

        # 合并阶段
        while i < len(left) and j < len(right):
            # 如果左边的数大于右边的数
            # 注意：left[i][1] 是该元素在原始 nums 中的索引
            if left[i][1] > right[j][1]:
                # 关键：当右边的元素被放入合并数组时，
                # 它意味着对于【左边剩余的所有元素】，都发现了一个比它们小的右侧元素。
                # 但更简单的做法是：当左边元素落位时，统计它跳过了多少个右边元素。
                merged.append(right[j])
                j += 1
            else:
                # 左边元素落位，此时 j 的数值就是右边比它小的元素个数
                self.res[left[i][0]] += j
                merged.append(left[i])
                i += 1

        # 处理剩余元素
        while i < len(left):
            self.res[left[i][0]] += j
            merged.append(left[i])
            i += 1

        while j < len(right):
            merged.append(right[j])
            j += 1

        return merged


# @lc code=end
