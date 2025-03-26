# PyTorch 代码示例集合

## 01 - 张量创建.py

```python
import torch
import numpy as np

#展示了普通数字和张量形式的区别
'''
它们都是10
张量具有更多特性并且支持更多高级运算（如自动求导、在 GPU 上运行、批量操作和深度学习框架无缝集成）
'''
print(10)
print(torch.tensor(10))


#展示了NumPy数组和张量形式的区别
'''
数据预处理常用 NumPy，深度学习模型训练用 PyTorch 张量。
'''
data_np=np.random.rand(2,3)
print(data_np)
print(torch.tensor(data_np))


#展示了张量的数据类型
'''
'''
data =[[10.,20,30],[20,30,40]]
print(data)
print(torch.tensor(data))



print(torch.Tensor([2])) # tensor([2.]) 输出中的 2. 表明该元素的数据类型为浮点数 维度为 [1]
print(torch.IntTensor(2,3)) #数据类型是torch.int32   维度为 [2,3] 传入的参数 2 和 3 是用来指定张量形状的
print(torch.DoubleTensor([2.3, 3.2])) #数据类型为 torch.float64 维度为 [2] 传入的是一个列表，PyTorch 会根据列表元素的情况创建一维张量，列表元素数量就是一维张量的长度。
print(torch.arange(0, 10, 2)) #创建一个等差数列0-10，步长为2
print(torch.linspace(0, 9, 10)) #创建一个等差数列0-9，共10个元素


# 展示了 PyTorch 中随机数生成和随机种子设置的过程
print(torch.randn(2, 3))
seed =torch.random.initial_seed()   #获取当前的随机种子值
print(seed)
torch.random.manual_seed(seed)
print(torch.randn(2, 3)) #设置相同的种子会产生相同的随机序列


# PyTorch 中创建特殊值张量的几种方法
data = torch.randn(4, 5)
print(data)
print(torch.zeros(2, 3))    #指定形状创建全0张量
print(torch.zeros_like(data))   # 创建一个与data形状相同的全0张量
print(torch.ones(2, 3))   #指定形状创建全1张量
print(torch.ones_like(data))  # 创建一个与data形状相同的全1张量


#在 PyTorch 中创建填充指定值的张量 填充值为100
print(torch.full([2, 3], 100))
print(torch.full_like(data, 100))

#输出数据类型
print(data.dtype)
print(data.type(torch.IntTensor).dtype)
print(data.int().dtype)
```



## 02 - 张量类型转换.py

```python
import torch


# 数据类型转换：PyTorch张量 → NumPy数组 展示了PyTorch张量和NumPy数组之间的转换，以及它们在数据共享方面的行为
torch.random.manual_seed(100)
data = torch.randn(2,3)
print(type(data))
data_numpy =data.numpy().copy() #转换为NumPy数组  在代码中使用 .copy() 的原因是为了演示完全独立的数据转换
print(type(data_numpy))
print(data)
data[0][1]= 100
print(data)     #数据修改：修改一个对象不影响另一个对象
print(data_numpy)


# 数据类型转换：NumPy数组 → PyTorch张量 展示了NumPy数组和PyTorch张量之间的转换，以及它们在数据共享方面的行为
data_tensor =torch.from_numpy(data_numpy.copy())
print(type(data_tensor))
print(data_tensor)
data_tensor[0][2]= 300
print(data_tensor)
print(data_numpy)


# 数据类型转换：PyTorch张量 → Python数值 展示了PyTorch张量和Python数值之间的转换
data_tensor =torch.tensor(data_numpy)
data_tensor[0][2]= 300
print(data_tensor)
print(data_numpy)

#对上述内容的解释
'''
# 当你想要完全独立的数据副本时
data_tensor = torch.tensor(data_numpy)
data_tensor[0][0] = 100  # 不会影响原始数据

# 当你想要节省内存并且需要数据同步时
data_tensor = torch.from_numpy(data_numpy)  # 共享内存 
data_tensor[0][0] = 100  # 会同步修改原始数据

'''





# 展示了如何从 PyTorch 张量中提取单个数值（标量值）
print(torch.tensor(30).item()) #创建一个包含单个值 30 的标量张量。 使用 .item() 方法将张量转换为 Python 数值
print(torch.tensor([30]).item()) #创建一个包含单个元素的一维张量 [30]
'''
.item() 方法的关键点是，它只能用于包含单个值的张量。 如果张量包含多个值，则会引发异常
# 这会引发错误，因为张量包含多个元素
tensor = torch.tensor([1, 2, 3])
value = tensor.item()  # 错误！
'''
```



## 03 - 张量的数值计算.py

```python
import torch

'''
展示了PyTorch中基本的张量操作
控制随机数生成
创建特定形状的随机张量
执行元素级运算的不同方法
'''
torch.random.manual_seed(10)
data1 = torch.randint(0, 10, [2, 3]) # 生成一个2x3的张量，元素取值范围为[0, 10)
torch.random.manual_seed(11)
data2 = torch.randint(0, 10, [2, 3])
# print(data)
# print(data.neg_()) #data.neg_()是张量的原地取负操作
# print(data)
print(data2)
print(data1)

# 两种完全等价的张量乘法方式
print(torch.mul(data1, data2))
print((data2 * data1))
```



## 04 - 张量的运算函数.py

```python
import torch

'''
这段代码展示了 PyTorch 中常用的张量操作 这些操作在深度学习中经常使用，比如：
聚合运算用于计算损失函数
指数运算用于 Softmax 激活函数
对数运算用于计算交叉熵损失
平方和平方根运算用于归一化操作
'''


torch.random.manual_seed(20)
data = torch.randint(0,10,[2,3],dtype=torch.float64)
print(data)

print(data.sum())       # 所有元素的总和
print(data.sum(dim=0))  # 按列求和（垂直方向）
print(data.sum(dim=1))  # 按行求和（水平方向）

print(data.mean())      # 所有元素的平均值
print(data.mean(dim=0)) # 每列的平均值
print(data.mean(dim=1)) # 每行的平均值

print(data.exp())           # 指数运算 e^x
print(torch.pow(data, 2))   # 平方运算 x^2
print(torch.pow(data, 0.5)) # 平方根运算 x^0.5
print(data.sqrt())          # 另一种平方根运算方式

print(data.log())    # 自然对数 ln(x)
print(data.log2())   # 以2为底的对数 log_2(x)
print(data.log10())  # 以10为底的对数 log_10(x)
```



## 05 - 张量的索引操作.py

```python
import torch

'''
展示了各种张量索引、切片操作和如何访问张量中的具体元素

应用场景：
批量数据处理
特征选择
数据切片
条件筛选
数据清洗：过滤异常值
'''

torch.random.manual_seed(10)
data = torch.randint(0, 10, [3, 4, 5])  # 创建3x4x5的三维随机整数张量
print(data)
# 基本索引
print(data[2,3])           # 第3层第4行的所有元素

# 列表索引
print(data[[0, 1], [1, 2]])  # 选择特定位置的元素

# 切片操作
print(data[0:3:2, :2])     # 隔行采样，并取前两列。0:3:2 的含义：从第1行到第4行，步长为2/ :2 的含义：取前两列 相当于 0:2
print(data[2:, :2])        # 从第3层开始，取前两列。 2: 的含义：从第3层开始 / :2 的含义：取前两列

# 条件索引
print(data[:, :, 2][data[:, :, 2] < 2])  # 选择第3列小于2的行。 data[:, :, 2] - 选择所有层、所有行的第3列/[data[:, :, 2] < 2] - 在上述结果中筛选出小于2的元素
print(data[:, 2] < 2)            # 条件判断结果 标识第3行中小于2的元素
print(data[:, data[1] > 5])      # 基于条件选择列 选择第2行中大于5的列


# 全选操作
print(data[0, :, :])      # 第1层的所有行和列
print(data[:, 0, :])      # 所有层的第1行
print(data[:, :, 2])      # 所有层所有行的第3列
print(data[1, 3, 4])     # 第2层第4行第5列的元素
```



## 06 - 张量的形状操作.py

```python
import torch

'''
应用场景：
调整数据维度和形状以匹配模型输入要求
批次处理时的数据重组
图像数据格式转换
特征图重排
序列数据的维度调整
注意力机制中的张量重排
检查和确保内存连续性
优化张量存储布局
'''
# 设置随机种子
torch.random.manual_seed(10)

# 创建随机张量
data = torch.randint(0, 10, [3, 4, 5, 6])
print("\n1. 初始数据:")
print(data)
print("初始形状:", data.shape)

# 重塑张量形状 2: 新张量的第一维大小/ 3: 新张量的第二维大小/ -1: 新张量的第三维大小。 不会修改原始张量
reshaped = data.reshape(2, 3,-1)
print("\n2. reshape操作:")
print("reshape后形状:", reshaped.size())

# dim=-1 表示在最后添加一个维度，dim=1表示在第二个维度添加一个维度，squeeze()表示压缩维度移除所有大小为1的维度
temp = data.unsqueeze(dim=-1).unsqueeze(dim=1)
print("\n3. 维度操作:")
print("添加维度后:", temp.shape)
print("压缩后:", temp.squeeze().shape)

# 转置 交换0,1维度，再交换1,2维度
print("\n4. 转置操作:")
print("两次转置后形状:", torch.transpose(torch.transpose(data, 0, 1),1,2).shape)

'''
维度重排就是改变张量各个维度的顺序，而不改变数据本身。
# 创建一个简单的3维张量
data = torch.arange(24).reshape(2, 3, 4)
print("原始张量:")
print(data)
print("原始形状:", data.shape)  # [2, 3, 4]

# 使用permute重排维度
permuted_data = data.permute(1, 0, 2)
print("\n重排后的张量:")
print(permuted_data)
print("重排后形状:", permuted_data.shape)  # [3, 2, 4]
'''

# 使用permute重排维度
print("\n5. Permute操作:")
print("torch.permute方式:", torch.permute(data, [1, 2, 3, 0]).shape)
print("tensor.permute方式:", data.permute([1, 2, 3, 0]).shape)

# 容错处理 检查张量是否在内存中连续存储，如果不是则调用contiguous()方法使其连续
data1 = torch.transpose(data, 0, 1)
print("\n6. 内存连续性处理:")
print("转置后是否连续:", data1.is_contiguous())

if data1.is_contiguous():
    result = data1.view(2, 4, -1) #view() 是 PyTorch 中用于改变张量形状的操作，2: 新张量的第一维大小/4: 新张量的第二维大小/-1 表示自动计算该维度大小
else:
    result = data1.contiguous().view(2, 4, -1)

print("view操作结果形状:", result.shape)
print("展平后形状:", data1.contiguous().view(-1).shape)
```



## 07 - 张量的拼接.py

```python
import torch

'''
应用场景：
特征拼接
模型输出合并
数据增强
多模态数据融合
'''
torch.random.manual_seed(10)

# 创建两个随机张量
data1 = torch.randint(0, 10, [3, 4, 5, 6])
data2 = torch.randint(0, 10, [3, 4, 3, 6])

# 打印原始形状
print("data1形状:", data1.shape)
print("data2形状:", data2.shape)

# 拼接并打印结果
result = torch.cat([data1, data2], dim=2)  #使用 torch.cat() 在第2维（dim=2）上拼接两个张量
print("拼接后形状:", result.shape)
```





## 08 - 案例 - 线性回归模型构建.py

```python
import torch
from torch.utils.data import TensorDataset  # 构造数据集对象
from torch.utils.data import DataLoader  # 数据加载器
from torch import nn  # nn模块中有平方损失函数和假设函数
from torch import optim  # optim模块中有优化器函数
from sklearn.datasets import make_regression  # 创建线性回归模型数据集
import matplotlib.pyplot as plt

# 设置matplotlib的中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def create_dataset():
    """创建训练数据集
    Returns:
        x (tensor): 输入特征
        y (tensor): 目标值
        coef (float): 真实系数
    """
    print("\n1. 生成训练数据...")
    x, y, coef = make_regression(
        n_samples=100,    # 样本数量
        n_features=1,     # 特征数量
        noise=10,         # 噪声水平
        coef=True,        # 返回真实系数
        bias=1.5,         # 偏置值
        random_state=0    # 随机种子
    )#生成线性回归数据
    
    print(f"   生成数据形状: x-{x.shape}, y-{y.shape}")
    print(f"   真实系数: {coef}, 真实偏置: 1.5")
    
    # 将构建数据转换为张量类型
    x = torch.tensor(x)
    y = torch.tensor(y)
    return x, y, coef

if __name__ == "__main__":
    print("开始线性回归数据生成和可视化过程...")
    # 生成的数据
    x, y, coef = create_dataset()
    
    print("\n2. 创建可视化图像...")
    # 绘制数据的真实的线性回归结果
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.5, label='原始数据点')
    print("   已绘制散点图")
    
    # 生成用于绘制真实线性关系的点
    x_line = torch.linspace(x.min(), x.max(), 1000)
    y_line = torch.tensor([v * coef + 1.5 for v in x_line])
    plt.plot(x_line, y_line, 'r-', label='真实关系线', linewidth=2)
    print("   已绘制真实关系线")
    
    # 添加图形元素
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('输入特征 (X)')
    plt.ylabel('目标值 (Y)')
    plt.title('线性回归数据分布及真实关系')
    plt.legend()
    
    plt.show()
    
    print("\n可视化过程完成!")
```

