# CheckFit.py

def check_model_fit(train_losses_D, train_losses_G, val_psnr_history, val_ssim_history, val_nc_history, epoch, logger):
    """
    判断模型是否欠拟合或过拟合
    每训练完5个epoch调用一次
    """
    if epoch < 5:
        # 训练初期，暂不判断
        return

    # 取最近5个epoch的数据
    recent_train_losses_D = train_losses_D[-5:]
    recent_train_losses_G = train_losses_G[-5:]
    recent_val_psnr = val_psnr_history[-5:]
    recent_val_ssim = val_ssim_history[-5:]
    recent_val_nc = val_nc_history[-5:]

    # 计算平均变化率
    def avg_change(data):
        changes = []
        for i in range(1, len(data)):
            changes.append(data[i] - data[i - 1])
        return sum(changes) / len(changes) if changes else 0

    avg_train_loss_D_change = avg_change(recent_train_losses_D)
    avg_train_loss_G_change = avg_change(recent_train_losses_G)
    avg_val_psnr_change = avg_change(recent_val_psnr)
    avg_val_ssim_change = avg_change(recent_val_ssim)
    avg_val_nc_change = avg_change(recent_val_nc)

    # 定义阈值
    improvement_threshold = 0.1  # 10% improvement

    overfitting = False
    underfitting = False

    # 检查过拟合：训练损失在下降，验证指标没有提升
    if (avg_train_loss_D_change < -improvement_threshold and
        avg_train_loss_G_change < -improvement_threshold and
        avg_val_psnr_change < improvement_threshold and
        avg_val_ssim_change < improvement_threshold and
        avg_val_nc_change < improvement_threshold):
        overfitting = True

    # 检查欠拟合：训练损失没有显著下降，验证指标没有提升
    if (abs(avg_train_loss_D_change) < improvement_threshold and
        abs(avg_train_loss_G_change) < improvement_threshold and
        abs(avg_val_psnr_change) < improvement_threshold and
        abs(avg_val_ssim_change) < improvement_threshold and
        abs(avg_val_nc_change) < improvement_threshold):
        underfitting = True

    if overfitting:
        logger.info("提示: 模型在最近5个epoch中过拟合。考虑使用正则化、增加数据或减少模型复杂度。")
    elif underfitting:
        logger.info("提示: 模型在最近5个epoch中欠拟合。考虑增加模型复杂度、减少正则化或增加训练时间。")