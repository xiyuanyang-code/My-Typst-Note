from math import gcd
from collections import deque


class Solution:
    def canMeasureWater(self, x: int, y: int, target: int) -> bool:
        return self.canMeasureWater_fast(x=x, y=y, target=target)

    def canMeasureWater_fast(self, x: int, y: int, target: int) -> bool:
        # for quicker solve
        if target > x + y:
            return False

        if x == 0 and y == 0:
            return target == 0

        if x == 0:
            return target % y == 0

        if y == 0:
            return target % x == 0

        return target % gcd(x, y) == 0

    def canMeasureWater_bfs(self, x: int, y: int, target: int) -> bool:
        # judge whether it can be measured directly
        if target > x + y:
            return False
        if target == 0:
            return True

        queue = deque()
        # initial state
        queue.append((0, 0))

        visited = set()
        visited.add((0, 0))

        while queue:
            a, b = queue.popleft()

            # when it is success
            if a == target or b == target or a + b == target:
                return True

            next_state = [(x, b), (a, y), (0, b), (a, 0)]

            # transitions
            pour_right = min(a, y - b)
            next_state.append((a - pour_right, b + pour_right))

            pour_left = min(b, x - a)
            next_state.append((a + pour_left, b - pour_left))

            # adding new state
            for state in next_state:
                if state not in visited:
                    visited.add(state)
                    queue.append(state)
        return False
