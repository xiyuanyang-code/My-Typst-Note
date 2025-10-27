"""
AI1804 算法设计与分析 - 第2次课上练习
问题2-4：医院诊室分配系统

学生姓名：杨希渊
学号：524531910015
"""


class Patient:
    """病人类"""

    def __init__(self, patient_id, name):
        self.patient_id = patient_id  # 挂号号码
        self.name = name

    def __str__(self):
        return f"病人{self.patient_id}({self.name})"

    def __repr__(self):
        return self.__str__()


class PatientNode:
    """病人链表节点"""

    def __init__(self, patient, next=None):
        self.patient: Patient = patient
        self.next = next


class HospitalSystem:
    """医院诊室分配系统"""

    def __init__(self, num_clinics):
        self.num_clinics = num_clinics
        self.clinics = [[] for _ in range(num_clinics)]  # k个诊室（数组）
        self.final_queue = []  # 最终取药队列

    def _get_clinics_with_minimum(self):
        min_clinic = self.clinics[0]
        min_clinic_length = len(self.clinics[0])
        for clinic in self.clinics:
            if len(clinic) < min_clinic_length:
                min_clinic = clinic
                min_clinic_length = len(clinic)
        return min_clinic

    def assign_patients_to_clinics(self, patient_list_head: PatientNode):
        """
        将病人分配到各个诊室，要求负载均衡且每个诊室内按号码有序

        算法要求：
        1. 负载均衡：选择当前病人数最少的诊室
        2. 有序维护：在选定诊室中按病人号码有序插入

        技术点：Linked List遍历 + Array操作 + 有序插入
        """
        current = patient_list_head
        while current:
            # find the minimum clinic
            chosen_clinic = self._get_clinics_with_minimum()
            self._insert_patient_sorted(clinic=chosen_clinic, patient=current)
            current = current.next

    def _insert_patient_sorted(self, clinic: list[Patient], patient: PatientNode):
        """
        将病人按号码顺序插入到诊室中（辅助方法）
        参数: clinic - 诊室列表, patient - 病人对象
        """
        if not clinic:
            clinic.append(patient.patient)
            return

        # using binary search to insert the element
        left = 0
        right = len(clinic) - 1
        while left <= right:
            mid = (left + right) // 2
            if clinic[mid].patient_id > patient.patient.patient_id:
                # insert left
                right = mid - 1
            elif clinic[mid].patient_id < patient.patient.patient_id:
                left = mid + 1
            else:
                raise (ValueError("Finding duplicate patient id!"))
        clinic.insert(left, patient.patient)

    def merge_clinic_queues(self):
        """
        合并所有诊室的队列为最终取药队列，要求使用分治法

        算法要求：
        1. 使用分治思想实现k路归并
        2. 最终队列按病人号码从小到大排序

        技术点：Divide & Conquer + Recursion + Array合并
        """
        return self._merge_k_queues_divide_conquer(queues=self.clinics, start=0, end=len(self.clinics)-1)

    def _merge_k_queues_divide_conquer(self, queues, start, end):
        """
        分治法合并队列范围[start, end]（辅助方法）
        """
        if start == end:
            return queues[start]
        elif start > end:
            # impossible
            return []
        else:
            pivot = (start + end) // 2
            left_merges = self._merge_k_queues_divide_conquer(
                queues=queues, start=start, end=pivot
            )
            right_merges = self._merge_k_queues_divide_conquer(
                queues=queues, start=pivot + 1, end=end
            )
            return self._merge_two_queues(left_merges, right_merges)

    def _merge_two_queues(self, queue1: list[Patient], queue2: list[Patient]):
        """
        合并两个有序队列（辅助方法）
        """
        result = []
        i = 0
        j = 0

        len1 = len(queue1)
        len2 = len(queue2)

        while i < len1 and j < len2:
            if queue1[i].patient_id <= queue2[j].patient_id:
                result.append(queue1[i])
                i += 1
            else:
                result.append(queue2[j])
                j += 1
        if i < len1:
            result.extend(queue1[i:])
        if j < len2:
            result.extend(queue2[j:])

        return result

    def process_hospital_queue(self, patient_list_head):
        """
        主处理流程 - 综合运用所有技术（已实现，展示整体流程）
        """
        print("开始医院排队系统处理...")

        # 步骤1: 统计病人信息（Linked List遍历）
        print("步骤1: 统计到达病人")
        total_patients = self._count_patients(patient_list_head)
        print(f"   总病人数: {total_patients}")

        # 步骤2: 分配病人到诊室（Linked List + Array + Sorting）
        print("步骤2: 分配病人到诊室（负载均衡+有序维护）")
        self.assign_patients_to_clinics(patient_list_head)

        # 显示分配结果
        for i, clinic in enumerate(self.clinics):
            patient_ids = [p.patient_id for p in clinic]
            print(f"   诊室{i+1}: {len(clinic)}人, 号码: {patient_ids}")

        # 步骤3: 合并诊室队列（Divide & Conquer + Recursion）
        print("步骤3: 合并诊室队列（k路归并）")
        self.merge_clinic_queues()

        final_ids = [p.patient_id for p in self.final_queue]
        print(f"   最终取药队列: {final_ids}")

        print("处理完成！")
        return {
            "total_patients": total_patients,
            "clinic_loads": [len(clinic) for clinic in self.clinics],
            "final_queue_size": len(self.final_queue),
            "is_final_sorted": all(
                self.final_queue[i].patient_id <= self.final_queue[i + 1].patient_id
                for i in range(len(self.final_queue) - 1)
            ),
        }

    def _count_patients(self, head):
        """统计病人总数（已实现的辅助方法）"""
        count = 0
        current = head
        while current:
            count += 1
            current = current.next
        return count


def main():
    """测试函数"""
    print("=== 问题2-4：医院诊室分配系统 ===")

    # 创建病人到达序列（号码有序但到达顺序随机）
    patients_data = [(5, "王五"), (2, "李二"), (8, "赵八"), (1, "张一"), (7, "孙七")]

    print(f"病人到达顺序: {[f'{id}({name})' for id, name in patients_data]}")

    # 构建链表
    head = PatientNode(Patient(patients_data[0][0], patients_data[0][1]))
    current = head
    for patient_id, name in patients_data[1:]:
        current.next = PatientNode(Patient(patient_id, name))
        current = current.next

    # 创建3个诊室的医院系统
    hospital = HospitalSystem(3)

    # 执行处理流程
    report = hospital.process_hospital_queue(head)

    print(f"\n=== 处理结果分析 ===")
    print(f"总病人数: {report['total_patients']}")
    print(f"各诊室负载: {report['clinic_loads']}")
    print(f"最终队列大小: {report['final_queue_size']}")
    print(f"最终队列是否有序: {report['is_final_sorted']}")

    # 验证负载均衡
    loads = report["clinic_loads"]
    max_load = max(loads)
    min_load = min(loads)
    print(
        f"负载均衡度: 最大负载{max_load}, 最小负载{min_load}, 差值{max_load - min_load}"
    )

    if max_load - min_load <= 1:
        print("✓ 负载均衡良好")
    else:
        print("✗ 负载不均衡")

    if report['is_final_sorted']:
        print("✓ 最终队列排序正确")
    else:
        print("✗ 最终队列排序错误")


if __name__ == "__main__":
    main()
