##################################################################################################
#    使用方法：
#    1. 将装甲板位置的实际值-预测值作为误差保存为.csv文件，具体代码如下：
#    main.cpp:
#    {
#       ......(other logic)
#       // 在下方代码中 armor3d.x/y/z 为实际装甲板中心，last_predict_x/y/z 为由上一帧预测的当前位置
#       // residuals_x/y/z为误差向量
#
#       std::vector<double> residuals_x, residuals_y, residuals_z, residuals_yaw;
#       double res_x = armor3d.tx - last_predict_x;
#       double res_y = armor3d.ty - last_predict_y;
#       double res_z = armor3d.tz - last_predict_z;
#       double res_yaw = armor3d.yaw - last_predict_yaw;
#
#       residuals_x.push_back(res_x);
#       residuals_y.push_back(res_y);
#       residuals_z.push_back(res_z);
#       residuals_yaw.push_back(res_yaw);
#
#       std::ofstream fout("your_csv_path.csv", std::ios::app);
#       fout << res_x << "," << res_y << "," << res_z << "," << res_yaw << std::endl;
#       fout.close();
#    }
#    2. 运行visualize_bias程序                                                                                         
##################################################################################################


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 修改数据文件路径
data = pd.read_csv('/home/wxy/NJU_RMCV/prediction_error.csv')

# 只选取数值型列
numeric_cols = ['res_x', 'res_y', 'res_z']

# 计算均值、方差
means = data[numeric_cols].mean()
variances = data[numeric_cols].var()
print('Mean Residuals (x, y, z):', means)
print('Variance Residuals (x, y, z,):', variances)

# 计算每个分量残差的绝对值均值（L1范数的均值）
mean_abs_res = data[['res_x', 'res_y', 'res_z']].abs().mean()
print('Mean Absolute Residuals (x, y, z):', mean_abs_res)

# 绘图
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 位置误差
axes[0].plot(data['res_x'], label='X Error')
axes[0].plot(data['res_y'], label='Y Error')
axes[0].plot(data['res_z'], label='Z Error')
axes[0].set_title('Position Prediction Error')
axes[0].set_ylabel('Error (m)')
axes[0].legend()
axes[0].grid(True)

# 误差分布直方图
axes[1].hist(data['res_x'], bins=30, alpha=0.7, label='X')
axes[1].hist(data['res_y'], bins=30, alpha=0.7, label='Y')
axes[1].hist(data['res_z'], bins=30, alpha=0.7, label='Z')
axes[1].set_title('Position Error Distribution')
axes[1].set_xlabel('Error (m)')
axes[1].legend()


plt.tight_layout()
plt.show()

# 打印统计信息
print("Position Error Statistics (m):")
print(f"X: mean={data['res_x'].mean():.4f}, std={data['res_x'].std():.4f}")
print(f"Y: mean={data['res_y'].mean():.4f}, std={data['res_y'].std():.4f}")
print(f"Z: mean={data['res_z'].mean():.4f}, std={data['res_z'].std():.4f}")
#print(f"Yaw: mean={data['res_yaw'].mean()*180/np.pi:.4f}°, std={data['res_yaw'].std()*180/np.pi:.4f}°")
