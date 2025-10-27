"""
AI1804 算法设计与分析 - 第2次课上练习
问题2-2：集合操作实现

学生姓名：___________
学号：___________
"""

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
        """
        从集合中删除元素x
        """
        # TODO: 实现删除操作
        found, index = self._binary_search(x)
        if found:
            self._data.pop(index)
            return True
        return False

    def find(self, x):
        """查找元素x是否在集合中"""
        # TODO: 实现查找操作
        found, _ = self._binary_search(x=x)
        return found

    def size(self):
        """返回集合大小"""
        return len(self._data)

    def to_list(self):
        """返回集合的有序列表表示"""
        return self._data.copy()
    
    def is_empty(self):
        """判断集合是否为空"""
        return len(self._data) == 0
    
    def __str__(self):
        """字符串表示"""
        return f"SortedSet({self._data})"

def main():
    """测试函数"""
    print("=== 问题2-2：集合操作实现 ===")
    
    # 创建集合并测试
    s = SortedArraySet()
    
    # 测试插入操作（包含重复元素）
    print("\n1. 插入测试：")
    test_data = [5, 2, 8, 1, 9, 3, 1, 2]  # 注意有重复元素
    print(f"插入数据: {test_data}")
    
    for x in test_data:
        s.insert(x)
        print(f"插入 {x} 后，集合: {s.to_list()}")
    
    print(f"最终集合: {s.to_list()}")
    print(f"集合大小: {s.size()}")
    
    # 测试查找操作
    print("\n2. 查找测试：")
    search_items = [1, 4, 8, 10]
    for item in search_items:
        found = s.find(item)
        print(f"查找 {item}: {'找到' if found else '未找到'}")
    
    # 测试删除操作
    print("\n3. 删除测试：")
    delete_items = [2, 10, 5]
    for item in delete_items:
        success = s.delete(item)
        print(f"删除 {item}: {'成功' if success else '失败'}")
        print(f"删除后集合: {s.to_list()}")

if __name__ == "__main__":
    main()
