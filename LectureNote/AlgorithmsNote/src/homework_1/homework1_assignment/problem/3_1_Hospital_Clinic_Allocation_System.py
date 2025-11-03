"""
Problem 3-1: Hospital Clinic Allocation System
EN: Assign patients to k clinics with balanced loads and per-clinic ascending order; when loads tie, choose the smaller clinic index. Finally, merge all clinic queues into one globally sorted queue.
CN: 将病人分配到 k 个诊室，保持负载均衡且各诊室内部按编号升序；负载相同时优先小编号诊室；最后将各诊室队列合并为一个全局升序队列。
"""
import math
from typing import List

# 假设病人编号就是 arrivals 列表中的 int 值

class HospitalSystem(object):

    def _insert_sorted(self, clinic: List[int], patient_id: int):
        """
        辅助方法：使用二分查找将病人编号有序地插入到诊室列表中。
        """
        if not clinic:
            clinic.append(patient_id)
            return

        left, right = 0, len(clinic) - 1
        insert_index = len(clinic)

        while left <= right:
            mid = (left + right) // 2
            if clinic[mid] > patient_id:
                # 插入点在左侧
                insert_index = mid
                right = mid - 1
            else:
                # clinic[mid] < patient_id，插入点在右侧
                left = mid + 1

        clinic.insert(insert_index, patient_id)

    def assign_patients_to_clinics(self, arrivals: List[int], k: int) -> List[List[int]]:
        """
        将到达序列中的病人分配到 k 个诊室，保证：
        1) 负载均衡：选择当前病人数最少的诊室。
        2) 诊室内按病人编号升序：使用有序插入。
        3) 负载相同时，分配到编号更小的诊室：通过从小到大遍历诊室实现。

        :type arrivals: List[int]
        :type k: int
        :rtype: List[List[int]]
        """
        if k <= 0:
            return []
        
        # 初始化 k 个诊室
        clinics: List[List[int]] = [[] for _ in range(k)]

        for patient_id in arrivals:
            min_load = math.inf
            chosen_clinic_index = -1

            for i in range(k):
                current_load = len(clinics[i])
                if current_load < min_load:
                    min_load = current_load
                    chosen_clinic_index = i

            # 将病人有序插入到选定的诊室
            if chosen_clinic_index != -1:
                self._insert_sorted(clinics[chosen_clinic_index], patient_id)

        return clinics

    def _merge_two_queues(self, queue1: List[int], queue2: List[int]) -> List[int]:
        """
        辅助方法：合并两个有序队列。
        """
        result = []
        i, j = 0, 0
        len1, len2 = len(queue1), len(queue2)

        while i < len1 and j < len2:
            if queue1[i] <= queue2[j]:
                result.append(queue1[i])
                i += 1
            else:
                result.append(queue2[j])
                j += 1
        
        # 添加剩余元素
        result.extend(queue1[i:])
        result.extend(queue2[j:])

        return result

    def _merge_k_queues_divide_conquer(self, queues: List[List[int]], start: int, end: int) -> List[int]:
        """
        辅助方法：分治法合并队列范围 [start, end]。
        """
        if start == end:
            return queues[start]
        elif start > end:
            return []
        else:
            mid = (start + end) // 2
            # 递归合并左半部分和右半部分
            left_merged = self._merge_k_queues_divide_conquer(queues, start, mid)
            right_merged = self._merge_k_queues_divide_conquer(queues, mid + 1, end)
            
            # 合并两个子结果
            return self._merge_two_queues(left_merged, right_merged)

    def merge_clinic_queues(self, queues: List[List[int]]) -> List[int]:
        """
        使用分治（两两归并）将 k 个已排序的诊室队列合并为一个全局升序队列。
        时间复杂度分析：
        - 共有 log k 层合并。
        - 每层合并操作的总复杂度为 O(n)，n 是所有病人总数。
        - 总时间复杂度为 O(n log k)。

        :type queues: List[List[int]]
        :rtype: List[int]
        """
        if not queues:
            return []
        
        return self._merge_k_queues_divide_conquer(queues, 0, len(queues) - 1)

    def process_hospital_queue(self, arrivals: List[int], k: int) -> List[int]:
        """
        主流程：分配 → 合并 → 返回最终全局队列。

        :type arrivals: List[int]
        :type k: int
        :rtype: List[int]
        """
        clinics = self.assign_patients_to_clinics(arrivals, k)

        # 步骤2: 合并诊室队列
        final_queue = self.merge_clinic_queues(clinics)
        
        return final_queue