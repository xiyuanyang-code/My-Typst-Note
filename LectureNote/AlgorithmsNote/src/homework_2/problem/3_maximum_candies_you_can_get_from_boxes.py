from typing import List
from collections import deque


class Solution:
    def maxCandies(
        self,
        status: List[int],
        candies: List[int],
        keys: List[List[int]],
        containedBoxes: List[List[int]],
        initialBoxes: List[int],
    ) -> int:
        total_box = len(status)
        candies_sum = 0
        # box_open_status: status
        box_contain_status = [0] * total_box
        visited_status = [0] * total_box

        # initialize
        queue = deque()
        for box_index in initialBoxes:
            box_contain_status[box_index] = 1
            if status[box_index] == 1:
                # the box is open, add to exploration queue
                queue.append(box_index)
        
        while queue:
            box_index = queue.popleft()
            visited_status[box_index] = 1
            
            # getting all the candies
            candies_sum += candies[box_index]

            updated_box = set()

            # get new box
            for new_box in containedBoxes[box_index]:
                box_contain_status[new_box] = 1
                updated_box.add(new_box)
            
            # get new keys
            for new_key in keys[box_index]:
                status[new_key] = 1
                updated_box.add(new_key)

            updates_box_list = list(updated_box)
            for updated_box in updates_box_list:
                if status[updated_box] == 1 and box_contain_status[updated_box] == 1:
                    queue.append(updated_box)
        return candies_sum