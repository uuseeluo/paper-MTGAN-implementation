import matplotlib
matplotlib.use('TkAgg')  # 设置使用 TkAgg 后端
import matplotlib.pyplot as plt

def read_data(file_name):
    """从文件中读取数据并返回两个列表，epochs和values。"""
    epochs = []
    values = []
    with open(file_name, 'r') as f:
        for line in f:
            # 使用逗号分隔
            epoch, value = line.strip().split(',')
            epochs.append(int(epoch))
            values.append(float(value))
    return epochs, values

# 读取数据
loss_D_epochs, loss_D_values = read_data('loss_D.txt')
loss_G_epochs, loss_G_values = read_data('loss_G.txt')
nc_after_epochs, nc_after_values = read_data('nc_after.txt')
nc_before_epochs, nc_before_values = read_data('nc_before.txt')
psnr_epochs, psnr_values = read_data('psnr.txt')
ssim_epochs, ssim_values = read_data('ssim.txt')
steganalysis_acc_epochs, steganalysis_acc_values = read_data('steganalysis_acc.txt')

# 创建 nc_after 和 nc_before 的折线图
plt.figure(figsize=(10, 6))
plt.plot(nc_after_epochs, nc_after_values, label='NC After', color='green')
plt.plot(nc_before_epochs, nc_before_values, label='NC Before', color='orange')
plt.title('NC After vs NC Before')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('nc_comparison.png')  # 保存 NC 比较图
plt.show()  # 显示 NC 比较图

# 创建独立的折线图
# 绘制 loss_D
plt.figure(figsize=(10, 6))
plt.plot(loss_D_epochs, loss_D_values, label='Loss D', color='red')
plt.title('Loss D')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_D.png')  # 保存 Loss D 图
plt.show()  # 显示 Loss D 图

# 绘制 loss_G
plt.figure(figsize=(10, 6))
plt.plot(loss_G_epochs, loss_G_values, label='Loss G', color='blue')
plt.title('Loss G')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_G.png')  # 保存 Loss G 图
plt.show()  # 显示 Loss G 图

# 绘制 PSNR
plt.figure(figsize=(10, 6))
plt.plot(psnr_epochs, psnr_values, label='PSNR', color='purple')
plt.title('PSNR')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.grid(True)
plt.tight_layout()
plt.savefig('psnr.png')  # 保存 PSNR 图
plt.show()  # 显示 PSNR 图

# 绘制 SSIM
plt.figure(figsize=(10, 6))
plt.plot(ssim_epochs, ssim_values, label='SSIM', color='brown')
plt.title('SSIM')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.grid(True)
plt.tight_layout()
plt.savefig('ssim.png')  # 保存 SSIM 图
plt.show()  # 显示 SSIM 图

# 绘制 Steganalysis Acc
plt.figure(figsize=(10, 6))
plt.plot(steganalysis_acc_epochs, steganalysis_acc_values, label='Steganalysis Acc', color='cyan')
plt.title('Steganalysis Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.grid(True)
plt.tight_layout()
plt.savefig('steganalysis_accuracy.png')  # 保存 Steganalysis Accuracy 图
plt.show()  # 显示 Steganalysis Accuracy 图