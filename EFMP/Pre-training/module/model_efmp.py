
from functools import partial

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode
from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed
from .bert_encoder import MultiModalBertEncoder

class FrequencyAwareMasking(nn.Module):
    """
    频率感知掩码生成模块，基于下采样图像的频率特性动态生成掩码。
    掩码数值：0 表示保留，1 表示移除。
    """
    def __init__(self, in_channels=1, patch_size=16, mask_ratio=0.75, reduction=0.0625, act_type='sigmoid'):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.reduction = reduction
        self.act_type = act_type

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

    def forward(self, img, x):
        N, L, D = x.shape
        assert img.shape[2] == img.shape[3] and img.shape[2] % self.patch_size == 0

        # 计算频率能量
        fft = torch.fft.fft2(img, norm='ortho')
        magnitude = torch.abs(fft)

        p = self.patch_size
        H, W = img.shape[2], img.shape[3]
        h = w = H // p
        patches = magnitude.unfold(2, p, p).unfold(3, p, p)
        patches = patches.reshape(N, self.in_channels, h * w, p, p)
        energy = patches.mean(dim=(3, 4))
        energy = energy.mean(dim=1)  # [N, h * w]

        # 生成动态频率权重
        freq_weights = self.freq_weight_conv(magnitude)  # [N, 1, H, W]
        freq_weights_patches = freq_weights.unfold(2, p, p).unfold(3, p, p)
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
        len_remove = int(L * self.mask_ratio)
        _, indices = torch.sort(energy, dim=1, descending=True)
        mask = torch.zeros(N, L, device=img.device)
        mask.scatter_(1, indices[:, :len_remove], 1)

        noise = torch.rand(N, L, device=img.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :int(L * (1 - self.mask_ratio))]

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

class efmp(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=768, decoder_depth=4, decoder_num_heads=6,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # 下采样到 112x112
        self.downsample = torchvision.transforms.Resize(
            (112, 112), interpolation=InterpolationMode.BICUBIC
        )

        # image encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # image decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, (patch_size)**2 * in_chans, bias=True)

        # Bert encoder
        self.bert_encoder = MultiModalBertEncoder()
        self.bert_mlp = nn.Linear(embed_dim, 768, bias=True)
        self.norm_pix_loss = norm_pix_loss

        # Frequency-aware masking for downsampled image
        self.freq_masking = FrequencyAwareMasking(in_channels=in_chans, patch_size=patch_size, mask_ratio=0.75)

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        p = self.patch_embed.patch_size[0] * 2
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def image_encoder(self, imgs, mask_ratio):
        # 下采样到 112x112
        imgs_down = self.downsample(imgs)
        
        # embed patches
        x = self.patch_embed(imgs)
        x = x + self.pos_embed[:, 1:, :]

        # frequency-aware masking on downsampled image
        x_masked, mask, ids_restore, ids_keep = self.freq_masking(imgs_down, x)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x_masked.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x_masked), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore, ids_keep

    def image_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]
        return x

    def forward_report_decoder(self, latent, ids_keep, caption_ids, labels, attention_mask, token_type_ids, weights):
        latent = self.bert_mlp(latent)
        gap_token = latent[:, 1:, :].mean(dim=1)
        gap_token = gap_token.unsqueeze(1)
        latent = latent[:, 1:, :]
        outputs = self.bert_encoder(latent, gap_token, caption_ids, labels, attention_mask, token_type_ids, weights)
        return outputs.loss

    def forward_loss(self, imgs, pred, mask):
        pixel_mask, _ = self.mask_2_pixel(mask)
        mask_pred_imgs = self.unpatchify(pred) * pixel_mask
        mask_imgs = imgs * pixel_mask
        mim_loss = F.mse_loss(mask_pred_imgs, mask_imgs, reduction='mean')
        return mim_loss

    def mask_2_pixel(self, mask):
        p = self.patch_embed.patch_size[0]
        mask = mask.reshape(shape=(mask.shape[0], int(mask.shape[1]**.5), int(mask.shape[1]**.5)))
        pixel_mask = torch.kron(mask, torch.ones((p, p)).cuda())
        pixel_mask = pixel_mask.unsqueeze(1).repeat(1, 3, 1, 1)
        return pixel_mask, None

    def forward(self, batch, mask_ratio=0.75):
        imgs = batch["image"]
        ids, labels, attention_mask, type_ids = batch["ids"], batch["labels"], batch["attention_mask"], batch["type_ids"]
        weights = batch["weights"]

        imgs = imgs.cuda()
        ids = ids.cuda()
        labels = labels.cuda()
        attention_mask = attention_mask.cuda()
        type_ids = type_ids.cuda()
        weights = weights.cuda()
        imgs = torchvision.transforms.Resize([224, 224], interpolation=InterpolationMode.BICUBIC)(imgs)

        latent, img_mask, ids_restore, ids_keep = self.image_encoder(imgs, mask_ratio)
        pred_img = self.image_decoder(latent, ids_restore)
        mim_loss = self.forward_loss(imgs, pred_img, img_mask)
        mlm_loss = self.forward_report_decoder(latent, ids_keep, ids, labels, attention_mask, type_ids, weights)
        return mim_loss, mlm_loss

def efmp(**kwargs):
    model = efmp(
        patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

