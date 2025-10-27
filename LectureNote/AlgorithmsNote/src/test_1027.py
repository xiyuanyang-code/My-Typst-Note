def two_sum(array: list[int], target: int):
    array_sorted = sorted(array)
    index_i = 0
    index_j = len(array) - 1
    while True:
        if array_sorted[index_i] + array_sorted[index_j] == target:
            # find the index
            index_i_origin = 0
            index_j_origin = 0
            for index, value in enumerate(array):
                if array_sorted[index_i] == value:
                    index_i_origin = index
                if array_sorted[index_j] == value:
                    index_j_origin = index
            return (index_i_origin, index_j_origin)
        elif array_sorted[index_i] + array_sorted[index_j] > target:
            index_j -= 1
        else:
            index_i += 1
        if index_i == index_j:
            return (-1, -1)


def max_sum_array(array: list[int]):
    length = len(array)
    if length == 1:
        return array[0]
    elif length == 2:
        return max(array[0], array[1], array[0] + array[1])
    split = length // 2
    # 0..split
    # split..length
    sum_left = max_sum_array(array=array[:split])
    sum_right = max_sum_array(array=array[split:])
    # compute medium max sum
    sum_medium_left = 0
    sum_medium_max_left = array[split - 1]
    sum_medium_right = 0
    sum_medium_max_right = array[split]
    for i in range(split - 1, -1, -1):
        sum_medium_left += array[i]
        sum_medium_max_left = max(sum_medium_max_left, sum_medium_left)
    for j in range(split, length):
        sum_medium_right += array[j]
        sum_medium_max_right = max(sum_medium_max_right, sum_medium_right)

    return max(sum_medium_max_left + sum_medium_max_right, sum_left, sum_right)


def find_peak_element(array: list[int]):
    for index, element in enumerate(array):
        if index == 0:
            continue
        if index == len(array) - 1:
            continue
        if element > array[index - 1] and element > array[index + 1]:
            return index
    return -1


def find_majority_element(arr: list[int]):
    """找到数组中出现次数大于n/2的元素"""
    # TODO: 实现多数元素查找算法
    dict_number = dict()
    target_length = len(arr) / 2
    for element in arr:
        if element not in dict_number.keys():
            dict_number[element] = 0
        else:
            dict_number[element] += 1
        if dict_number[element] > target_length:
            return element


class SortedArraySet:
    def __init__(self):
        """初始化集合"""
        self._data = []  # 存储数据的有序列表

    def _binary_search(self, x):
        """二分查找元素x的位置，返回(found, index)"""
        # TODO: 实现二分查找算法
        length = self.size()
        left_bound = 0
        right_bound = length - 1

        while left_bound <= right_bound:
            target_index = (left_bound + right_bound) // 2

            if self._data[target_index] == x:
                return (True, target_index)
            elif self._data[target_index] > x:
                right_bound = target_index - 1
            else:  # self._data[target_index] < x
                left_bound = target_index + 1

        return (False, left_bound)

    def insert(self, x):
        """向集合中插入元素x"""
        # TODO: 实现插入操作
        found, index = self._binary_search(x)
        if not found:
            self._data.insert(index, x)

    def delete(self, x):
        """从集合中删除元素x，返回True如果成功，False如果元素不存在"""
        # TODO: 实现删除操作
        found, index = self._binary_search(x)
        if found:
            self._data.remove(x)

    def find(self, x):
        """查找元素x是否在集合中"""
        # TODO: 实现查找操作
        found, _ = self.find(x=x)
        return found
        # 其他方法...

    def size(self):
        """返回集合大小"""
        return len(self._data)

    def to_list(self):
        """返回集合的有序列表表示"""
        return self._data.copy()


def merge_sort_with_inversion_count(arr):
    """
    归并排序并统计逆序对数量
    参数: arr - 待排序的数组
    返回: (sorted_arr, inversion_count) - 排序后的数组和逆序对数量
    """

    def merge(left, right):
        """合并两个有序数组，并统计跨越逆序对"""
        # TODO: 实现合并逻辑，同时统计逆序对
        index_i = 0
        index_j = 0
        merge_count = 0
        new_array = []
        while index_i < len(left) and index_j < len(right):
            if left[index_i] > right[index_j]:
                merge_count += len(left) - index_i
                new_array.append(right[index_j])
                index_j += 1
            else:
                new_array.append(left[index_i])
                index_i += 1
        new_array.extend(left[index_i:])
        new_array.extend(right[index_j:])
        return new_array, merge_count

    def merge_sort_helper(arr):
        """递归实现归并排序并统计逆序对"""
        # TODO: 实现归并排序的递归逻辑

        if len(arr) == 1:
            return arr, 0
        if len(arr) == 0:
            return arr, 0

        pivot = len(arr) // 2
        merge_left, count_left = merge_sort_helper(arr[:pivot])
        merge_right, count_right = merge_sort_helper(arr[pivot:])
        merge_new, merge_count = merge(merge_left, merge_right)
        return merge_new, merge_count + count_left + count_right

    return merge_sort_helper(arr)


def count_inversions_naive(arr):
    """
    朴素方法统计逆序对（用于验证结果）
    时间复杂度: O(n^2)
    """
    count = 0
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] > arr[j]:
                count += 1
    return count


if __name__ == "__main__":
    # print(two_sum([1,2,3,4,5,6,2,3,4,-2], 10))
    # print(max_sum_array([1,-3,2,1,-1]))
    # s = SortedArraySet()
    # test = [5, 2, 8, 1, 9, 3,1,2]
    # for x in test:
    #     s.insert(x)
    # print(f"集合: {s.to_list()}")

    # 测试代码
    arr = [3, 1, 4, 2, 4, 5, 6, 7, 3, 3, 5, 6, 7, 8, 2]
    sorted_arr, inv_count = merge_sort_with_inversion_count(arr)
    print(sorted_arr, inv_count)

    print(count_inversions_naive(arr))
    # print(f"排序后: {sorted_arr}, 逆序对数量: {inv_count}")
