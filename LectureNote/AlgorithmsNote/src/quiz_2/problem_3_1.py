"""
AI1804 算法设计与分析
第3次课上练习：哈希表与优先队列

问题3-1：字符串基数排序

请在TODO标记处填写代码
"""


def counting_sort_by_char(arr, char_index):
    """
    按指定字符位置进行稳定的计数排序

    参数:
        arr: 字符串数组
        char_index: 要排序的字符位置（0表示第一个字符）

    返回:
        排序后的数组

    提示: 使用计数数组（256个桶）+ 累积和 + 稳定填充
    """
    n = len(arr)
    if n == 0:
        return arr

    count = [0] * 256
    for s in arr:
        ch = ord(s[char_index]) if char_index < len(s) else 0
        count[ch] += 1

    for i in range(1, 256):
        count[i] += count[i - 1]

    result = [None] * n
    for i in range(n - 1, -1, -1):
        s = arr[i]
        ch = ord(s[char_index])
        pos = count[ch] - 1         
        result[pos] = s
        count[ch] -= 1

    return result


def string_radix_sort(arr):
    """
    字符串基数排序

    参数:
        arr: 等长字符串数组

    返回:
        排序后的数组

    时间复杂度: O(d * n)，d是字符串长度
    """
    if not arr:
        return []

    # TODO: 从最后一个字符到第一个字符，依次调用counting_sort_by_char
    single_string_length = len(arr[0])
    for i in range(single_string_length - 1, -1, -1):
        arr = counting_sort_by_char(arr, char_index=i)
    return arr


# ========== 测试代码 ==========

if __name__ == "__main__":
    print("=" * 60)
    print("问题3-1：字符串基数排序 - 测试")
    print("=" * 60)

    # 测试1
    arr1 = ["abc", "abd", "aba", "aaa", "bcd"]
    sorted_arr1 = string_radix_sort(arr1)
    print(f"\n测试1:")
    print(f"  输入: {arr1}")
    print(f"  输出: {sorted_arr1}")
    print(f"  期望: ['aaa', 'aba', 'abc', 'abd', 'bcd']")

    expected1 = ["aaa", "aba", "abc", "abd", "bcd"]
    if sorted_arr1 == expected1:
        print("  ✓ 测试1通过")
    else:
        print("  ✗ 测试1失败")

    # 测试2
    arr2 = ["dog", "cat", "bat", "rat"]
    sorted_arr2 = string_radix_sort(arr2)
    print(f"\n测试2:")
    print(f"  输入: {arr2}")
    print(f"  输出: {sorted_arr2}")
    print(f"  期望: ['bat', 'cat', 'dog', 'rat']")

    expected2 = ["bat", "cat", "dog", "rat"]
    if sorted_arr2 == expected2:
        print("  ✓ 测试2通过")
    else:
        print("  ✗ 测试2失败")

    print("\n" + "=" * 60)
