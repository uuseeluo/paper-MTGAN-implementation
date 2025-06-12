# config.py
import matplotlib

matplotlib.use('Agg')


config = {
    'video_dir': './Videodata/UCF-101/**/*.avi',  # Directory pattern for cover video files
    'save_dir': 'results',                         # Directory to save results
    'log_dir': 'logs',                             # Directory for logs
    'metrics_dir': 'metrics',                      # Directory to save metrics files
    'batch_size': 8,                               # Adjust batch_size to save memory
    'num_epochs': 20,                              # Number of training epochs
    'learning_rate': 0.0005,                        # Learning rate
    'beta1': 0.9,                                   # Adam optimizer beta1
    'beta2': 0.999,                                 # Adam optimizer beta2
    'lambda_recon': 10.0,                           # Reconstruction loss weight
    'lambda_sec': 150.0,                            # Secret extraction loss weight
    'lambda_sec_after': 2.0,
    'lambda_perc': 1.0,                             # Perceptual loss weight
    'lambda_adv': 1.0,                              # Adversarial loss weight
    'lambda_reg': 1e-4,                             # Regularization loss weight
    'lambda_steg_adv': 0.1,
    'lambda_gp': 10.0,                              # Gradient penalty weight
    'lambda_steg': 1.0,
    'lambda_D_reg': 1e-4,
    'fusion_type': 'additive',                      # Fusion type: 'additive', 'multiplicative', 'attention'
    'attention_heads': 4,                           # Number of attention heads (if using attention fusion)
    'num_workers': 2,                               # Number of DataLoader workers
    'validation_size': 100,                          # Number of samples for validation
    'save_interval': 1,                             # Save models every 'save_interval' epochs
    'n_visualize': 5,                               # Number of batches to visualize during validation
    'n_visualize_sample': 3,                        # Number of samples to visualize per batch
    'scheduler_step_size': 5,                      # Scheduler step size
    'scheduler_gamma': 0.3,                         # Scheduler decay factor
    'noise_std': 0.02,                              # 噪声参数
    'use_amp': True,                                # 是否启用混合精度训练
    'check_fit_interval': 5,
    'weight_decay': 0.0001,  # 根据需要调整
    'num_visualization_samples': 4,
    'secret_augmentation': True,
    'aug_brightness': 0.2,
    'aug_contrast': 0.2,
    'aug_saturation': 0.2,
    'aug_hue': 0.1,

}

