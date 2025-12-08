from typing import List


def get_max_happiness(temperatures: List[int]):
    len_days = len(temperatures)
    dp_in = [None] * len_days
    dp_not_in = [None] * len_days

    dp_in_ans = [None] * len_days
    dp_not_in_ans = [None] * len_days

    # update:
    dp_in[0] = temperatures[0]
    dp_in_ans[0] = [0]

    dp_not_in[0] = 0
    dp_not_in_ans[0] = []

    dp_in[1] = temperatures[1]
    dp_in_ans[1] = [1]

    dp_not_in[1] = max(temperatures[0], 0)
    dp_not_in_ans[1] = [0] if temperatures[0] >= 0 else []

    for i in range(2, len_days):
        dp_in[i] = temperatures[i] + max(dp_in[i - 2], dp_not_in[i - 2])
        if dp_in[i - 2] > dp_not_in[i - 2]:
            dp_in_ans[i] = dp_in_ans[i - 2]
            dp_in_ans[i].append(i)
        else:
            dp_in_ans[i] = dp_not_in_ans[i - 2]
            dp_in_ans[i].append(i)

        dp_not_in[i] = max(dp_in[i - 1], dp_not_in[i - 1])
        if dp_in[i - 1] > dp_not_in[i - 1]:
            dp_not_in_ans[i] = dp_in_ans[i - 1]
        else:
            dp_not_in_ans[i] = dp_not_in_ans[i - 1]

    if dp_in[len_days - 1] > dp_not_in[len_days - 1]:
        result = dp_in_ans[len_days - 1]
    else:
        result = dp_not_in_ans[len_days - 1]

    return list(set(range(len_days)) - set(result))


if __name__ == "__main__":
    samples = {
        "sample_1": {
            "temperatures": [3, 2, 1],
            "max_happiness": 4,
            "study_days_0_based": [1],
        },
        "sample_2": {
            "temperatures": [-1, -2, -3],
            "max_happiness": 0,
            "study_days_0_based": [0, 1, 2],
        },
        "sample_3": {
            "temperatures": [4, 5, -1, 6],
            "max_happiness": 11,
            "study_days_0_based": [0, 2],
        },
        "sample_4": {
            "temperatures": [1, 2, 3, 4],
            "max_happiness": 6,
            "study_days_0_based": [0, 2],
        },
        "sample_5": {
            "temperatures": [2, -5, 3, -2, 7],
            "max_happiness": 12,
            "study_days_0_based": [1, 3],
        },
    }

    for i in range(1, 6):
        sample_name = f"sample_{i}"
        assert (
            get_max_happiness(temperatures=samples[sample_name]["temperatures"])
            == samples[sample_name]["study_days_0_based"]
        )
    print("All test passed")
