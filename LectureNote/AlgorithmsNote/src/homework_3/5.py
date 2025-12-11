from typing import List


def get_max_happiness(temperatures: List[int]):
    len_days = len(temperatures)
    dp_in = [None] * len_days
    dp_not_in = [None] * len_days

    dp_in[0] = temperatures[0]
    dp_not_in[0] = 0

    if len_days == 1:
        return max(dp_in[0],dp_not_in[0])

    dp_in[1] = temperatures[1]
    dp_not_in[1] = max(temperatures[0], 0)

    if len_days == 2:
        return max(dp_in[1],dp_not_in[1])

    for i in range(2, len_days):
        dp_in[i] = temperatures[i] + max(dp_in[i - 2], dp_not_in[i - 2])
        dp_not_in[i] = max(dp_in[i - 1], dp_not_in[i - 1])
    return max(dp_in[len_days - 1],dp_not_in[len_days - 1])


if __name__ == "__main__":
    n = int(input())
    temp = list(map(int, input().split()))
    print(get_max_happiness(temp))

