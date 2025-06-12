import torch


def compute_gradient_penalty(discriminator, real_samples, fake_samples, device, lambda_gp):
    """Calculates the gradient penalty loss for WGAN GP"""
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    outputs = discriminator(interpolates)

    # 检查输出是否为元组
    if isinstance(outputs, tuple) and len(outputs) == 2:
        validity, _ = outputs
    else:
        validity = outputs

    fake = torch.ones(validity.size(), device=device, requires_grad=False)
    gradients = torch.autograd.grad(
        outputs=validity,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()
    return penalty