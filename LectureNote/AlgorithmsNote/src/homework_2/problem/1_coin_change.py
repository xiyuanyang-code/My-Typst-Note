from typing import List


class Solution(object):
    def coinChange(self, coins: List[int], amount: int):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        if amount == 0:
            return 0

        dp = [1e5] * (amount + 1)
        dp[0] = 0
        for i in range(1, amount + 1):
            for coin in coins:
                if i - coin >= 0:
                    dp[i] = min(dp[i], dp[i-coin] + 1)
        return dp[amount] if dp[amount] != 1e5 else -1
