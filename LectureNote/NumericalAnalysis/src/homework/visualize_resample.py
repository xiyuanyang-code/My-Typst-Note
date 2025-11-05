import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# 导入resample函数
from hm2 import resample

def visualize_resample(series, tgt_length, title="Resample Visualization"):
    """
    可视化resample函数的工作原理
    """
    # 原始数据
    src_length = len(series)
    x_old = np.linspace(0, src_length - 1, num=src_length)
    y_old = np.array(series)
    
    # 重采样后的数据
    resampled_series = resample(series, tgt_length)
    x_new = np.linspace(0, src_length - 1, num=tgt_length)
    y_new = np.array(resampled_series)
    
    # 创建图表
    plt.figure(figsize=(10, 6))
    
    # 绘制原始数据点
    plt.scatter(x_old, y_old, color='red', s=50, label='Original Data Points', zorder=5)
    
    # 绘制原始数据的线性插值曲线
    if src_length > 1:
        interpolator = interp1d(x_old, y_old, kind="linear")
        x_smooth = np.linspace(0, src_length - 1, num=1000)
        y_smooth = interpolator(x_smooth)
        plt.plot(x_smooth, y_smooth, 'r--', alpha=0.7, label='Linear Interpolation')
    
    # 绘制重采样后的数据点
    plt.scatter(x_new, y_new, color='blue', s=30, label=f'Resampled Points (n={tgt_length})', zorder=5)
    
    # 连接重采样点的线
    plt.plot(x_new, y_new, 'b-', alpha=0.5)
    
    # 设置图表属性
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, (x, y) in enumerate(zip(x_old, y_old)):
        plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', color='red')
    
    for i, (x, y) in enumerate(zip(x_new, y_new)):
        plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0,-15), ha='center', color='blue', fontsize=8)
    
    plt.tight_layout()
    return plt

def demonstrate_resample():
    """
    演示resample函数的几种情况
    """
    # 示例1: 扩展序列
    series1 = [1.0, 3.0, 2.0, 5.0]
    tgt_length1 = 10
    plt1 = visualize_resample(series1, tgt_length1, "Example 1: Expanding Series")
    plt1.savefig("resample_example1_expanding.png", dpi=150, bbox_inches='tight')
    plt1.show()
    
    # 示例2: 压缩序列
    series2 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    tgt_length2 = 4
    plt2 = visualize_resample(series2, tgt_length2, "Example 2: Compressing Series")
    plt2.savefig("resample_example2_compressing.png", dpi=150, bbox_inches='tight')
    plt2.show()
    
    # 示例3: 复杂波形
    series3 = [1,3,4,2]
    tgt_length3 =3
    plt3 = visualize_resample(series3, tgt_length3, "Example 3: Complex Waveform")
    plt3.savefig("resample_example3_waveform.png", dpi=150, bbox_inches='tight')
    plt3.show()

def generate_complex_testcases():
    """
    生成更复杂的测试用例
    """
    complex_testcases = [
        # 测试浮点数精度
        {
            "series": [0.1, 0.2, 0.3],
            "tgt_length": 5,
            "gt": [0.1, 0.15, 0.2, 0.25, 0.3],
            "description": "Floating point precision test"
        },
        # 测试负数
        {
            "series": [-5.0, -2.0, 1.0, 4.0],
            "tgt_length": 7,
            "gt": [-5.0, -3.5, -2.0, -0.5, 1.0, 2.5, 4.0],
            "description": "Negative numbers test"
        },
        # 测试大数组压缩
        {
            "series": list(range(0, 101, 10)),  # [0, 10, 20, ..., 100]
            "tgt_length": 5,
            "gt": [0.0, 25.0, 50.0, 75.0, 100.0],
            "description": "Large array compression test"
        },
        # 测试大数组扩展
        {
            "series": [1.0, 2.0],
            "tgt_length": 20,
            "gt": [1.0, 1.0526315789473684, 1.1052631578947367, 1.1578947368421053, 
                   1.2105263157894737, 1.263157894736842, 1.3157894736842106, 
                   1.3684210526315788, 1.4210526315789473, 1.4736842105263157, 
                   1.5263157894736843, 1.5789473684210527, 1.631578947368421, 
                   1.6842105263157894, 1.736842105263158, 1.789473684210526, 
                   1.8421052631578947, 1.894736842105263, 1.9473684210526314, 2.0],
            "description": "Large array expansion test"
        },
        # 测试包含零的数组
        {
            "series": [0, 5, 0, -5, 0],
            "tgt_length": 9,
            "gt": [0.0, 2.5, 5.0, 2.5, 0.0, -2.5, -5.0, -2.5, 0.0],
            "description": "Array with zeros test"
        },
        # 测试单元素数组扩展
        {
            "series": [42.0],
            "tgt_length": 5,
            "gt": [42.0, 42.0, 42.0, 42.0, 42.0],
            "description": "Single element expansion test"
        },
        # 测试相同值数组
        {
            "series": [7.0, 7.0, 7.0, 7.0],
            "tgt_length": 10,
            "gt": [7.0] * 10,
            "description": "Constant value array test"
        },
        # 测试非常小的数组扩展
        {
            "series": [1.0, 10.0],
            "tgt_length": 1,
            "gt": [1.0],
            "description": "Two elements compressed to one"
        }
    ]
    
    return complex_testcases

def run_complex_tests():
    """
    运行复杂测试用例
    """
    from hm2 import test
    
    complex_testcases = generate_complex_testcases()
    
    print("Running complex test cases...\n")
    
    all_passed = True
    for i, test_case in enumerate(complex_testcases, 1):
        print(f"Test {i}: {test_case['description']}")
        passed = test(test_case["series"], test_case["tgt_length"], test_case["gt"])
        if not passed:
            all_passed = False
        print()
    
    if all_passed:
        print("🎉 All complex tests passed!")
    else:
        print("❌ Some tests failed.")
        
    return all_passed

if __name__ == "__main__":
    # 运行复杂测试
    run_complex_tests()
    
    # 创建可视化示例（可选）
    demonstrate_resample()