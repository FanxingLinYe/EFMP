import torch
import torch.nn as nn
import torchvision
from torchvision.transforms.functional import InterpolationMode
import matplotlib.pyplot as plt
import numpy as np
import os

class FrequencyAwareMasking(nn.Module):
    """
    频率感知掩码生成模块，基于图像的频率特性动态生成掩码。
    掩码数值：0 表示保留，1 表示移除。
    使用卷积网络生成动态频率权重，关注病灶区域的高频特征。
    """
    def __init__(self, in_channels=1, patch_size=16, mask_ratio=0.75, reduction=0.0625, act_type='sigmoid'):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.reduction = reduction  # 固定为 0.0625，适合单通道 CXR 图像
        self.act_type = act_type

        # 下采样变换调整为与 patch_size 匹配
        self.downsample = torchvision.transforms.Resize(
            (patch_size * 14, patch_size * 14),
            interpolation=InterpolationMode.BICUBIC
        )

        # 卷积模块生成动态频率权重
        attention_channel = max(int(in_channels * reduction), 16)
        self.freq_weight_conv = nn.Sequential(
            nn.Conv2d(in_channels, attention_channel, 1, bias=False),
            nn.BatchNorm2d(attention_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(attention_channel, 1, 1, bias=True)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, img, random_mask=False):
        N, C, H, W = img.shape
        assert H == W and H % self.patch_size == 0

        if random_mask:
            L = (H // self.patch_size) * (W // self.patch_size)
            len_keep = int(L * (1 - self.mask_ratio))
            noise = torch.rand(N, L, device=img.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            ids_keep = ids_shuffle[:, :len_keep]
            mask = torch.ones(N, L, device=img.device)
            mask[:, :len_keep] = 0  # 0 表示保留，1 表示移除
            mask = torch.gather(mask, dim=1, index=ids_restore)
            return mask, ids_restore, ids_keep

        # 计算频率能量
        fft = torch.fft.fft2(img, norm='ortho')
        magnitude = torch.abs(fft)

        p = self.patch_size
        h = w = H // p
        patches = magnitude.unfold(2, p, p).unfold(3, p, p)
        patches = patches.reshape(N, C, h * w, p, p)
        energy = patches.mean(dim=(3, 4))
        energy = energy.mean(dim=1)  # [N, h * w]

        # 生成动态频率权重
        freq_weights = self.freq_weight_conv(magnitude)  # [N, 1, H, W]
        freq_weights_patches = freq_weights.unfold(2, p, p).unfold(3, p, p)  # [N, 1, h, w, p, p]
        freq_weights_patches = freq_weights_patches.reshape(N, 1, h * w, p * p)
        freq_weights = freq_weights_patches.mean(dim=3)  # [N, 1, h * w]
        freq_weights = freq_weights.reshape(N, h * w)  # [N, h * w]
        if self.act_type == 'sigmoid':
            freq_weights = torch.sigmoid(freq_weights)
        elif self.act_type == 'tanh':
            freq_weights = 1 + torch.tanh(freq_weights)
        else:
            raise NotImplementedError

        # 结合频率能量和动态权重
        energy = energy * freq_weights  # [N, h * w]
        energy = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)

        # 根据 mask_ratio 选择移除的 patch
        L = h * w
        len_remove = int(L * self.mask_ratio)
        _, indices = torch.sort(energy, dim=1, descending=True)  # 高权重优先移除
        mask = torch.zeros(N, L, device=img.device)
        mask.scatter_(1, indices[:, :len_remove], 1)  # 1 表示移除

        noise = torch.rand(N, L, device=img.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :int(L * (1 - self.mask_ratio))]

        mask = torch.gather(mask, dim=1, index=ids_restore)
        return mask, ids_restore, ids_keep

def visualize_mask_multi_columns(original_img, filename="cxr_masked_comparison.png", mask_ratios=[0.75], mask_color=[1.0, 0.0, 0.0]):
    """
    可视化多列图像：原始图像、随机掩码图像、动态频率掩码图像。
    掩码区域使用指定颜色（默认红色）显示。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original_img = original_img.to(device)

    # 将单通道图像转换为三通道（RGB），以支持彩色掩码
    N, C, H, W = original_img.shape
    if C == 1:
        original_img_rgb = original_img.repeat(1, 3, 1, 1)
    else:
        original_img_rgb = original_img

    # 计算列数：1（原始）+ 1（随机掩码）+ 1（动态掩码）
    num_cols = 3
    fig, axes = plt.subplots(1, num_cols, figsize=(5 * num_cols, 5))

    # 显示原始图像
    img_np_orig = original_img[0].cpu().numpy().squeeze()
    img_np_orig = (img_np_orig - img_np_orig.min()) / (img_np_orig.max() - img_np_orig.min() + 1e-8)
    if C == 1:
        axes[0].imshow(img_np_orig, cmap='gray')
    else:
        axes[0].imshow(img_np_orig.transpose(1, 2, 0))
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # 显示随机掩码图像
    freq_masking = FrequencyAwareMasking(in_channels=C, patch_size=16, mask_ratio=mask_ratios[0], reduction=0.0625).to(device)
    mask_random, _, _ = freq_masking(original_img, random_mask=True)
    h = w = H // 16
    mask_random = mask_random.reshape(N, h, w)
    pixel_mask_random = torch.kron(mask_random, torch.ones((16, 16), device=device))
    pixel_mask_random = pixel_mask_random.unsqueeze(1).repeat(1, 3, 1, 1)
    masked_img_random = original_img_rgb.clone()
    mask_color_tensor = torch.tensor(mask_color, device=device).view(1, 3, 1, 1)
    masked_img_random = masked_img_random * (1 - pixel_mask_random) + pixel_mask_random * mask_color_tensor
    img_np_random = masked_img_random[0].cpu().numpy().transpose(1, 2, 0)
    img_np_random = (img_np_random - img_np_random.min()) / (img_np_random.max() - img_np_random.min() + 1e-8)
    axes[1].imshow(img_np_random)
    axes[1].set_title("Random Masked")
    axes[1].axis('off')

    # 显示动态频率掩码
    mask_freq, _, _ = freq_masking(original_img, random_mask=False)
    mask_freq = mask_freq.reshape(N, h, w)
    pixel_mask_freq = torch.kron(mask_freq, torch.ones((16, 16), device=device))
    pixel_mask_freq = pixel_mask_freq.unsqueeze(1).repeat(1, 3, 1, 1)
    masked_img_freq = original_img_rgb.clone()
    masked_img_freq = masked_img_freq * (1 - pixel_mask_freq) + pixel_mask_freq * mask_color_tensor
    img_np_freq = masked_img_freq[0].cpu().numpy().transpose(1, 2, 0)
    img_np_freq = (img_np_freq - img_np_freq.min()) / (img_np_freq.max() - img_np_freq.min() + 1e-8)
    axes[2].imshow(img_np_freq)
    axes[2].set_title("Dynamic Frequency Masked")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    from PIL import Image
    img_path = "6ee2157f-a0a756cc-95dfa431-50f5a24a-3b5c1c82.jpg"
    # img_path = "a788e14d-66bca74c-e111283f-f5da2271-fc34b51c.jpg"
    # img_path = "248f179a-c68a800c-35209997-38618805-e13b7dec.jpg"
    # img_path = "d7b5a64a-eb36c15a-e0d63ddb-01deb656-901c18e1.jpg"
    if os.path.exists(img_path):
        img = Image.open(img_path).convert('L')
        img = torch.tensor(np.array(img)).unsqueeze(0).unsqueeze(0).float()
        img = torchvision.transforms.Resize((224, 224))(img)
        visualize_mask_multi_columns(
            img,
            filename="FREAM.png",
            mask_ratios=[0.75],
            mask_color=[1.0, 1.0, 1.0]  # 白色掩码
        )
        print("可视化已保存为 FREAM.png")
    else:
        print("请将 'your_cxr_image.png' 替换为你的 CXR 图像路径。")
