"""
AI1804 算法设计与分析 - 第2次课上练习
问题2-1.3：寻找峰值元素

学生姓名：杨希渊
学号：524531910015
"""

def find_peak_element(arr):
    """
    找到数组中的峰值元素，返回其索引 (迭代/非递归实现)
    
    时间复杂度：O(log n)
    空间复杂度：O(1)
    """
    length = len(arr)
    if length == 0:
        return -1
    if length == 1:
        return 0
    
    left = 0
    right = length - 1 
    
    while left < right:
        mid = left + (right - left) // 2
        if arr[mid] < arr[mid + 1]:
            left = mid + 1
        elif arr[mid] > arr[mid + 1]:
            right = mid

    return left

def find_peak_naive(arr):
    """
    朴素方法查找峰值（用于验证结果）
    时间复杂度: O(n)
    """
    if not arr:
        return -1
    
    n = len(arr)
    
    # 检查第一个元素
    if n == 1 or arr[0] > arr[1]:
        return 0
    
    # 检查最后一个元素
    if arr[n-1] > arr[n-2]:
        return n-1
    
    # 检查中间元素
    for i in range(1, n-1):
        if arr[i] > arr[i-1] and arr[i] > arr[i+1]:
            return i
    
    return -1  # 理论上不会到达这里

def verify_peak(arr, index):
    """
    验证给定索引是否是峰值
    """
    if index == -1 or index >= len(arr):
        return False
    
    n = len(arr)
    
    # 检查左邻居
    left_ok = (index == 0) or (arr[index] > arr[index-1])
    
    # 检查右邻居
    right_ok = (index == n-1) or (arr[index] > arr[index+1])
    
    return left_ok and right_ok

def main():
    """测试函数"""
    print("=== 问题2-1.3：寻找峰值元素 ===")
    
    # 测试用例1：基础测试
    print("\n1. 基础测试：")
    arr1 = [1, 2, 3, 1]
    print(f"数组: {arr1}")
    
    naive_result = find_peak_naive(arr1)
    divide_result = find_peak_element(arr1)
    
    print(f"朴素方法结果: 索引 {naive_result}")
    if naive_result != -1:
        print(f"  峰值: {arr1[naive_result]}")
    
    print(f"分治方法结果: 索引 {divide_result}")
    if divide_result != -1:
        print(f"  峰值: {arr1[divide_result]}")
    
    # 验证结果
    if divide_result != -1 and verify_peak(arr1, divide_result):
        print("✓ 分治方法找到有效峰值")
    else:
        print("✗ 分治方法结果无效")
    
    # 测试用例2：多个峰值
    print("\n2. 多个峰值测试：")
    arr2 = [1, 3, 2, 4, 1]
    print(f"数组: {arr2}")
    
    naive_result2 = find_peak_naive(arr2)
    divide_result2 = find_peak_element(arr2)
    
    print(f"朴素方法结果: 索引 {naive_result2}")
    if naive_result2 != -1:
        print(f"  峰值: {arr2[naive_result2]}")
    
    print(f"分治方法结果: 索引 {divide_result2}")
    if divide_result2 != -1:
        print(f"  峰值: {arr2[divide_result2]}")
    
    if divide_result2 != -1 and verify_peak(arr2, divide_result2):
        print("✓ 分治方法找到有效峰值")
    else:
        print("✗ 分治方法结果无效")
    
    # 测试用例3：单调递增
    print("\n3. 单调递增测试：")
    arr3 = [1, 2, 3, 4, 5]
    print(f"数组: {arr3}")
    
    divide_result3 = find_peak_element(arr3)
    print(f"分治方法结果: 索引 {divide_result3}")
    if divide_result3 != -1:
        print(f"  峰值: {arr3[divide_result3]}")
    
    if divide_result3 != -1 and verify_peak(arr3, divide_result3):
        print("✓ 分治方法找到有效峰值")
        print("  (单调递增数组的峰值应该是最后一个元素)")
    else:
        print("✗ 分治方法结果无效")
    
    # 测试用例4：单调递减
    print("\n4. 单调递减测试：")
    arr4 = [5, 4, 3, 2, 1]
    print(f"数组: {arr4}")
    
    divide_result4 = find_peak_element(arr4)
    print(f"分治方法结果: 索引 {divide_result4}")
    if divide_result4 != -1:
        print(f"  峰值: {arr4[divide_result4]}")
    
    if divide_result4 != -1 and verify_peak(arr4, divide_result4):
        print("✓ 分治方法找到有效峰值")
        print("  (单调递减数组的峰值应该是第一个元素)")
    else:
        print("✗ 分治方法结果无效")
    
    # 测试用例5：单个元素
    print("\n5. 单个元素测试：")
    arr5 = [42]
    print(f"数组: {arr5}")
    
    divide_result5 = find_peak_element(arr5)
    print(f"分治方法结果: 索引 {divide_result5}")
    if divide_result5 != -1:
        print(f"  峰值: {arr5[divide_result5]}")
    
    if divide_result5 == 0:
        print("✓ 单个元素正确识别为峰值")
    else:
        print("✗ 单个元素应该是峰值")

    # test case 6
    arr6 = [1,1,1,2,2,2,4,4,4,5,5,5]

    # todo about this strange testcase?
    # print(find_peak_naive(arr6))
    # print(find_peak_element(arr6))
    
    print("\n=== 算法要求 ===")
    print("时间复杂度: O(log n) - 每次排除一半搜索空间")
    print("空间复杂度: O(log n) - 递归调用栈深度")
    print("核心技巧: 分治+二分思想，利用峰值必定存在的性质")
    print("关键洞察: 峰值必定存在，可以根据邻居关系确定搜索方向")

if __name__ == "__main__":
    main()
