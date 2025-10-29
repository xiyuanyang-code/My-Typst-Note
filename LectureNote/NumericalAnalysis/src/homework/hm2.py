import numpy as np
from scipy.interpolate import interp1d


def resample(series, tgt_length):
    if len(series) == 1:
        return series * tgt_length
    if len(series) == tgt_length:
        return series
    if tgt_length == 1:
        return [series[0]]

    src_length = len(series)
    x_old = np.linspace(0, src_length - 1, num=src_length)
    y_old = np.array(series)

    # 分段线性插值
    interpolator = interp1d(x_old, y_old, kind="linear")
    x_new = np.linspace(0, src_length - 1, num=tgt_length)
    resampled_series = interpolator(x_new)
    return resampled_series.tolist()


def test(series, tgt_length, gt):
    print(f"Testcase for {series, tgt_length}")
    answer = resample(series=series, tgt_length=tgt_length)
    if len(answer) == len(gt) and np.allclose(answer, gt, rtol=1e-5, atol=1e-8):
        print("✅: Testcase Passed")
        return True
    else:
        print("❌: Testcase Failed")
        print(f"GT: {gt}")
        print(f"Answer: {answer}")
        return False


if __name__ == "__main__":
    TESTCASES = [
        # ID 1
        {
            "series": [5.0, 15.0],
            "tgt_length": 6,
            "gt": [5.0, 7.0, 9.0, 11.0, 13.0, 15.0],
        },
        # ID 2
        {
            "series": [100.0, 200.0, 300.0, 400.0],
            "tgt_length": 4,
            "gt": [100.0, 200.0, 300.0, 400.0],
        },
        # ID 3
        {"series": [1.0, 2.0, 3.0, 4.0], "tgt_length": 1, "gt": [1.0]},
        # ID 4
        {
            "series": [1.0, 0.0, 1.0],
            "tgt_length": 7,
            "gt": [1.0, 2 / 3, 1 / 3, 0.0, 1 / 3, 2 / 3, 1.0],
        },
        # ID 5
        {"series": [4.0, 4.0, 4.0, 4.0], "tgt_length": 8, "gt": [4.0] * 8},
        {"series": [1.0, 3.0, 4.0], "tgt_length": 2, "gt": [1.0, 4.0]}
    ]
    for test_case in TESTCASES:
        test(test_case["series"], test_case["tgt_length"], test_case["gt"])
