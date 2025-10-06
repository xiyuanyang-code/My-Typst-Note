# utility functions


def counting_leading_zeros(s: str):
    count = 0
    for char in s:
        if char == "0":
            count += 1
        else:
            break
    return count


def format_to_sci(num_str: str):
    """
    将字符串数值转为 (digits_str, exponent)
    不使用 decimal，仅字符串操作
    """
    num_str = num_str.strip().lower()

    if "e" in num_str:
        base, exp_part = num_str.split("e")
        exp_shift = int(exp_part)
    elif "E" in num_str:
        base, exp_part = num_str.split("E")
        exp_shift = int(exp_part)
    else:
        base, exp_part = num_str, "0"
        exp_shift = 0

    sign = 1
    if base.startswith("-"):
        sign = -1
        base = base[1:]
    elif base.startswith("+"):
        base = base[1:]

    # 分离整数和小数
    if "." in base:
        int_part, frac_part = base.split(".")
    else:
        int_part, frac_part = base, ""

    # 去掉前导零
    int_part = int_part.lstrip("0")
    if int_part == "":
        int_part = "0"

    # get all the digits part
    digits_str = int_part + frac_part

    if digits_str == len(digits_str) * "0":
        return digits_str, 0
    else:
        digits_str = digits_str.lstrip("0")

    if int_part != "0" and int_part != "":
        # 并不是只存在 0
        exp_add = len(int_part) - 1
    else:
        exp_add = -(1 + counting_leading_zeros(frac_part))

    exponent = exp_shift + exp_add

    return (("-" + digits_str if sign == -1 else digits_str), exponent)


def abs_diff_in_sci(num1: str, num2: str):
    """
    计算两个字符串数字的差的绝对值，返回 (digits_str, exponent)。
    约定：format_to_sci 返回 (digits_str, exponent)，其中 digits_str 是不带小数点的有效数字字符串，
    原数值 = int(digits_str) * 10 ** (exponent - (len(digits_str)-1))

    例: "3.145" -> ("3145", 0)
    """

    # 将输入转为 (digits, exponent)
    digits1, exp1 = format_to_sci(num1)
    digits2, exp2 = format_to_sci(num2)

    # 处理符号
    neg1 = digits1.startswith("-")
    neg2 = digits2.startswith("-")
    d1 = digits1[1:] if neg1 else digits1
    d2 = digits2[1:] if neg2 else digits2

    # 快速处理零（如果某个 digits 是 "0"）
    if d1 == "0" and d2 == "0":
        return "0", 0

    # 计算 p = exponent - (len(digits)-1) ，使数 = int(digits) * 10^p
    len1 = len(d1)
    len2 = len(d2)
    p1 = exp1 - (len1 - 1)
    p2 = exp2 - (len2 - 1)

    # 把两个数都缩放到相同的 10^p_common 基准上（p_common = min(p1,p2)）
    p_common = min(p1, p2)
    scale1 = p1 - p_common  # >= 0
    scale2 = p2 - p_common  # >= 0

    int1 = int(d1) * (10**scale1)
    int2 = int(d2) * (10**scale2)

    # 恢复符号
    if neg1:
        int1 = -int1
    if neg2:
        int2 = -int2

    diff_int = abs(int1 - int2)

    # 若差为 0，统一返回 ("0", 0)
    if diff_int == 0:
        return "0", 0

    diff_digits = str(diff_int)
    # 转回 (digits, exponent)：使得 diff_digits * 10^(exp_out - (len(diff_digits)-1)) == diff_int * 10^p_common
    exp_out = p_common + (len(diff_digits) - 1)

    return diff_digits, exp_out


def get_significant_figure(ref, est):
    """计算实数估计值 est 相对于实数参考值 ref 的有效数字位数

    注：应支持负数情况。【可选】支持 1.5e-3 科学计数法形式的输入

    Args:

        ref (str): 实数参考值的字符串形式

        est (str): 实数估计值的字符串形式

    Returns:

        n (int): 有效数字位数

    """
    ref_digits, ref_exponent = format_to_sci(ref)
    est_digits, est_exponent = format_to_sci(est)
    # print(ref_digits, ref_exponent, est_digits, est_exponent)

    # calculate m: x
    x = int(ref_exponent)

    # calculate m-n+1: y
    diff_digits, diff_exponent = abs_diff_in_sci(ref, est)
    # print(diff_digits, diff_exponent)
    if diff_digits == "0":
        # todo if ref == est?
        return -1

    # get the first element of diff_digits
    diff_first = int(diff_digits[0])
    if diff_first < 5:
        y = diff_exponent + 1
    elif diff_first == 5:
        if diff_digits.rstrip("0") == "5":
            # only 5:
            y = diff_exponent + 1
        else:
            y = diff_exponent + 2
    else:
        y = diff_exponent + 2

    return max(x - y + 1, 0)


if __name__ == "__main__":
    assert get_significant_figure("3.1415926", "3.1415") == 4
    assert get_significant_figure("0.001234", "0.00123") == 3
    assert get_significant_figure("1.234e-3", "1.23e-3") == 3
    assert get_significant_figure("-3.1415926", "-3.1415 ") == 4
    assert get_significant_figure("-0.001234", "-0.00123") == 3
    assert get_significant_figure("-1.234e-3", "-1.23e-3") == 3
    assert get_significant_figure("-3.1415926", "-3.1415") == 4
    assert get_significant_figure("2.2530", "2.3000") == 2
    assert get_significant_figure("3.14", "3.1416") == 3

    # 长小数测试：300位有效数字
    ref = "3." + "14159" * 60  # 3.14159... 重复到大约 300 位
    est = ref[:-1]  # 少掉最后一位
    assert get_significant_figure(ref, est) == 299

    # 极小数：1e-200 和近似值
    ref = "1e-200"
    est = "1.00001e-200"
    assert get_significant_figure(ref, est) == 5

    # 极大数：1000位整数
    ref = "9" * 1000  # 999...999 (1000位9)
    est = "9" * 999 + "8"  # 最后一位错
    assert get_significant_figure(ref, est) == 999

    # 相差极大的情况（有效数字 = 0）
    # print(get_significant_figure("1.234e50", "9.876e20"))
    assert get_significant_figure("1.234e50", "9.876e20") == 0
    assert get_significant_figure("12345678901234567890", "12345678") == 0
    assert get_significant_figure("0.0001234", "98765") == 0
    assert get_significant_figure("999999999999", "0.000000000001") == 0

    print("All Test Passed!")
