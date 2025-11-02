# 深度学习第一次作业

杨希渊 524531910015

## Problem 1
什么是维数灾难? 请举例说明维数灾难可能带来的问题。

维数灾难是指随着数据维度的增加，空间体积呈指数级增长，从而导致数据变得极其稀疏，并引发一系列计算、统计和模型构建上的问题。

维数灾难可能带来的问题：

- 数据稀疏性：在高维空间中，数据点分布变得极度稀疏。高维球体随机采样极容易采到表面。
- 计算复杂性：高维空间中搜索最近邻等一些操作的计算复杂度会随维度指数级增长。
- 模型过拟合：高维函数拟合需要采样指数多的样本，但实际中可能没有足够的数据。



## Problem 2

请绘制出 $d = 5, 10, 20$ 时 d 维球体体积 $V(d)$ 随半径 $r \in [0, 1]$ 的变化曲线, 分析维数 $d$ 增大时的影响。

$d$ 维球体 $B_d(r)$ 的体积 $V_d(r)$ 可以写成如下形式：
$$
V_d(r) = C_d r^d
$$
求解系数可得具体的表达式：
$$
V_d(r) = \frac{\pi^{d/2}}{\Gamma\left(\frac{d}{2} + 1\right)} r^d
$$
其中 $\Gamma(z) = \int_0^{\infin} t^{z-1} e^{-t} \text{d}t$, $\Gamma(n) = (n-1)!, n \in \mathbb{N}^+$.

### Code

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

def volume_of_d_ball(r, d):
    """
    Calculate the volume V_d(r) of a d-dimensional ball with radius r.
    
    Formula: V_d(r) = [pi^(d/2) / Gamma(d/2 + 1)] * r^d
    
    Parameters:
    r (float or array-like): Radius of the ball.
    d (int): Dimension.
    
    Returns:
    float or array-like: Volume of the d-dimensional ball.
    """
    # Calculate volume constant C_d = pi^(d/2) / Gamma(d/2 + 1)
    C_d = (np.pi**(d / 2)) / gamma(d / 2 + 1)
    
    # Volume V_d(r) = C_d * r^d
    V_d_r = C_d * (r**d)
    
    return V_d_r

def plot_d_ball_volume_curves():
    """
    Plot the volume curves of balls in dimensions d=5, 10, 20 as a function of radius.
    """
    # 1. Define dimensions to plot
    dimensions = [5, 10, 20]
    
    # 2. Define radius range (from 0 to 1.5 to better observe growth trends)
    r_values = np.linspace(0, 1, 300)
    
    # 3. Initialize plotting
    plt.figure(figsize=(10, 6))
    plt.title('Volume $V_d(r)$ of High-Dimensional Balls vs Radius $r$', fontsize=16)
    plt.xlabel('Radius $r$', fontsize=14)
    plt.ylabel('Volume $V_d(r)$', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 4. Loop to calculate and plot each curve
    for d in dimensions:
        # Calculate volume values for the corresponding dimension
        V_d_values = volume_of_d_ball(r_values, d)
        
        # Plot the curve with label
        plt.plot(r_values, V_d_values, 
                 label=f'$d = {d}$', 
                 linewidth=2.5)

    # 5. Add annotations and legend
    
    # Mark unit radius (r=1) position
    plt.axvline(x=1.0, color='gray', linestyle=':', linewidth=1.5)
    plt.text(1.02, plt.ylim()[1] * 0.9, '$r=1$ (unit radius)', 
             rotation=0, color='gray')
    
    plt.legend(fontsize=12, loc='upper left')
    plt.ylim(bottom=0) # Ensure volume starts from 0
    plt.savefig("./LectureNote/DeepLearning/homework/d_ball.pdf")

# Execute visualization function
plot_d_ball_volume_curves()
```

### Result

![image-20251102230517808](/Users/xiyuanyang/Library/Application Support/typora-user-images/image-20251102230517808.png)

## Problem 3

蒙特卡洛方法的收敛阶是 $O(n^{-\frac{1}{2}})$, 与维数 $d$ 无关，即使采样数目 $n$ 已经很大了，但我们前面在使用神经网络学习一个高维二次函数时在某些点的泛化误差依然很大，应该如何理解这个现象?

蒙特卡洛方法自适应地在单位球的球壳附近采了大量的样本点，导致模型在积分意义下的泛化误差确实很小（因为球壳的体积占比很大），但是在高维空间数据中，球体的体积主要集中在表面，因此靠近球心的采样点哪怕概率密度很高但是积分出来的被采样的概率非常低，因此这部分点未被充分采样，虽然 $L_2$ 误差很小但是 $L_{\infin}$ 的误差很大，**即带来很大的泛化误差**。
