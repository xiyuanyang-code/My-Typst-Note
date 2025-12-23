from typing import List


def build_wall(B: List[str]):
    k = len(B)
    n = len(B[0])
    inf = float("inf")

    num_states = 1 << (2 * k)

    # dp[c][r][mask]
    dp = [[[inf] * num_states for _ in range(k)] for _ in range(n + 1)]
    parent = [[[None] * num_states for _ in range(k)] for _ in range(n + 1)]

    # 初始状态
    dp[0][0][0] = 0

    for c in range(n):
        for r in range(k):
            for mask in range(num_states):
                if dp[c][r][mask] == inf:
                    continue

                # 1. 确定下一个网格坐标和状态的基础掩码
                if r == k - 1:
                    next_c, next_r = c + 1, 0
                    # 换列时：原本的下一列(高k位)变成当前列(低k位)，新的下一列全清空
                    shifted_mask = mask >> k
                else:
                    next_c, next_r = c, r + 1
                    shifted_mask = mask

                # 2. 检查当前格子 (c, r)
                is_occupied = (mask >> r) & 1
                is_dirt = B[r][c] == "#"

                if is_dirt or is_occupied:
                    # 情况 1: 跳过
                    if dp[next_c][next_r][shifted_mask] > dp[c][r][mask]:
                        # do relaxations
                        dp[next_c][next_r][shifted_mask] = dp[c][r][mask]
                        parent[next_c][next_r][shifted_mask] = (c, r, mask, None)
                else:
                    # 情况 2: 必须填

                    # 动作 '1': 放置碎石 (1x1x1)
                    # 不改变 mask 任何位，直接传给 next (如果是换列则用 shifted_mask)
                    if dp[next_c][next_r][shifted_mask] > dp[c][r][mask] + 1:
                        dp[next_c][next_r][shifted_mask] = dp[c][r][mask] + 1
                        parent[next_c][next_r][shifted_mask] = (c, r, mask, "1")

                    # 动作 'R': 水平放置 (1x2) 占领 (c, r) 和 (c+1, r)
                    if c + 1 < n and B[r][c + 1] == ".":
                        # 占领下一列的第 r 行，即 mask 的第 r + k 位
                        new_mask_val = mask | (1 << (r + k))
                        # 如果换列，需右移
                        final_mask = new_mask_val >> k if r == k - 1 else new_mask_val
                        if dp[next_c][next_r][final_mask] > dp[c][r][mask]:
                            dp[next_c][next_r][final_mask] = dp[c][r][mask]
                            parent[next_c][next_r][final_mask] = (c, r, mask, "R")

                    # 动作 'D': 垂直放置 (1x2) 占领 (c, r) 和 (c, r+1)
                    if r + 1 < k and B[r + 1][c] == "." and not ((mask >> (r + 1)) & 1):
                        new_mask_val = mask | (1 << (r + 1))
                        # 垂直块不可能跨列，所以 r 必不等于 k-1，直接传给下一行
                        if dp[next_c][next_r][new_mask_val] > dp[c][r][mask]:
                            dp[next_c][next_r][new_mask_val] = dp[c][r][mask]
                            parent[next_c][next_r][new_mask_val] = (c, r, mask, "D")

    # 回溯
    ans_p = []
    curr_c, curr_r, curr_mask = n, 0, 0
    while parent[curr_c][curr_r][curr_mask] is not None:
        prev_c, prev_r, prev_mask, action = parent[curr_c][curr_r][curr_mask]
        if action:
            ans_p.append((prev_c, prev_r, action))
        curr_c, curr_r, curr_mask = prev_c, prev_r, prev_mask

    return ans_p[::-1]


def parse_map_from_input(k):
    """
    k: 地图的行数
    """
    grid = []
    for _ in range(k):
        line = list(input().strip())
        grid.append(line)
    return grid


if __name__ == "__main__":
    k = 5
    grid = parse_map_from_input(k)
    print(build_wall(grid))
