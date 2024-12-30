import matplotlib

matplotlib.use('Agg')  # 使用 'Agg' 后端，避免 GUI 相关问题
import matplotlib.pyplot as plt

# 定义读取 accuracy_results_resnet.txt 文件的函数
def read_results(filename):
    noise_ratios = []
    accuracies = []

    with open(filename, 'r') as f:
        # 跳过第一行（表头）
        next(f)

        # 逐行读取数据
        for line in f:
            if line.strip():  # 确保跳过空行
                noise_ratio, accuracy = line.split()
                noise_ratios.append(float(noise_ratio))  # 将噪声率转为浮点数
                accuracies.append(float(accuracy))  # 将准确率转为浮点数

    return noise_ratios, accuracies

# 读取结果
noise_ratios, accuracies = read_results('accuracy_results_mil_resnet_attention.txt')

# 生成折线图
plt.figure(figsize=(8, 6))
plt.plot(noise_ratios, accuracies, marker='o', linestyle='-', color='b', label='Test Accuracy')

# 设置图表标题和标签
plt.title('Test Accuracy vs. Noise Ratio', fontsize=14)
plt.xlabel('Noise Ratio (%)', fontsize=12)
plt.ylabel('Test Accuracy (%)', fontsize=12)

# 设置纵坐标范围
plt.ylim(50, 100)

# 设置横坐标范围，留出一些距离
plt.xlim(-5, 50)  # 设置 x 轴范围从 -5 到 50，留出一点距离

# 获取起点和终点的坐标
start_x, start_y = noise_ratios[0], accuracies[0]  # 起点坐标
end_x, end_y = noise_ratios[-1], accuracies[-1]  # 终点坐标

# 起点标注
plt.annotate(f'({start_x}, {start_y:.2f}%)',
             xy=(start_x, start_y),
             xytext=(start_x + 2, start_y + 3),  # 调整文字位置
             arrowprops=dict(facecolor='black', arrowstyle="->", lw=1),
             fontsize=10)

# 终点标注
plt.annotate(f'({end_x}, {end_y:.2f}%)',
             xy=(end_x, end_y),
             xytext=(end_x - 5, end_y - 5),  # 调整文字位置
             arrowprops=dict(facecolor='black', arrowstyle="->", lw=1),
             fontsize=10)

# 添加虚线映射到坐标轴
plt.axvline(x=start_x, color='gray', linestyle='--', linewidth=1)  # 起点的虚线
plt.axhline(y=start_y, color='gray', linestyle='--', linewidth=1)  # 起点的虚线
plt.axvline(x=end_x, color='gray', linestyle='--', linewidth=1)  # 终点的虚线
plt.axhline(y=end_y, color='gray', linestyle='--', linewidth=1)  # 终点的虚线

# 添加网格和图例
plt.grid(True)
plt.legend()

# 保存图表为 PDF 文件
plt.savefig('resnet_mil_attention.pdf')  # 保存为 PDF 图片
