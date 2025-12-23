from typing import List
import random




class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next: ListNode = next

    def get_length(self):
        current_length = 0
        current_node = self
        while self:
            current_length += 1
            current_node = current_node.next


def merge(l1, l2):
    """
    原地合并两个有序链表，并返回合并后的头和尾
    """
    dummy = ListNode(0)
    curr = dummy
    while l1 and l2:
        if l1.val < l2.val:
            curr.next, l1 = l1, l1.next
        else:
            curr.next, l2 = l2, l2.next
        curr = curr.next

    curr.next = l1 if l1 else l2

    # 找到合并后的尾部，用于后续连接
    while curr.next:
        curr = curr.next
    return dummy.next, curr


def split(head, n):
    """
    断开链表，使前 n 个节点独立，并返回第 n+1 个节点的引用
    """
    for i in range(n - 1):
        if not head:
            break
        head = head.next

    if not head:
        return None

    next_part = head.next
    head.next = None  # 切断连接
    return next_part


def build_node(input):
    current_node = None
    if len(input) == 0:
        return None
    else:
        node = ListNode(input[0])
        current_node = node
        for i in range(1, len(input)):
            current_node.next = ListNode(input[i])
            current_node = current_node.next
    return node


def sort_list_top_down(head: ListNode) -> ListNode:
    length = head.get_length()
    if length == 0 or length == 1:
        return head

    def merge_list_sort(left: int, right: int, left_begin_node: ListNode):
        mid = (left + right) // 2
        current_node = left_begin_node
        for _ in range(mid):
            current_node = current_node.next
        right_begin_node = current_node

        left_merge = merge_list_sort(left, mid - 1, left_begin_node=left_begin_node)
        right_merge = merge_list_sort(mid, right, right_begin_node)
        new_node = merge(left_merge, right_merge)
        return new_node

    return merge_list_sort(0, length - 1, head)


def sort_list_bottom_up(head: ListNode) -> ListNode:
    if not head or not head.next:
        return head

    length = 0
    curr = head
    while curr:
        length += 1
        curr = curr.next

    dummy = ListNode(0)
    dummy.next = head

    step = 1
    while step < length:
        prev = dummy
        curr = dummy.next

        while curr:
            left = curr
            right = split(left, step)
            remaining = split(right, step)
            merged_head, merged_tail = merge(left, right)
            prev.next = merged_head
            prev = merged_tail
            curr = remaining

        step *= 2

    return dummy.next


def sort_list(input: List[int]):
    head = build_node(input=input)
    # using the bottom up methods
    sorted_head = sort_list_bottom_up(head=head)

    result = []
    curr = sorted_head
    while curr:
        result.append(curr.val)
        curr = curr.next
    return result


if __name__ == "__main__":
    input_str = input()
    if input_str.strip():
        input_list = [int(x.strip()) for x in input_str.split(",")]
        sorted_numbers = sort_list(input_list)
        print(",".join(map(str, sorted_numbers)))