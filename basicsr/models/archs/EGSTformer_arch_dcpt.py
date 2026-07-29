"""
HOGformer with DCPT Support
============================
Modified to support degradation classification pre-training (DCPT)

Key Changes:
1. Added hook mechanism in forward() to extract encoder features
2. Compatible with degradation_classification_pretrain_model.py
3. Extracts 4 levels of encoder features: [level1, level2, level3, latent]

Usage:
    For DCPT pretraining: model(inp_img, hook=True)
    For normal training: model(inp_img, hook=False) or model(inp_img)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange

#########################################################################
# ========== Import DynamicTanh ========== #
#########################################################################
try:
    from .dynamic_tanh import DynamicTanh
except ImportError:
    try:
        from dynamic_tanh import DynamicTanh
    except:
        print("[Warning] DynamicTanh not found, using WithBias LayerNorm")

#########################################################################
# ========== Sparse Prompt Module (SPM) Components ========== #
#########################################################################

class HOGGuidedPromptGenBlock(nn.Module):
    """HOG-Guided Sparse Prompt Generation Block"""
    def __init__(self, prompt_dim=128, prompt_len=5, prompt_size=96, 
                 lin_dim=192, num_expert=2, n_bins=9):
        super(HOGGuidedPromptGenBlock, self).__init__()
        
        self.prompt_param = nn.Parameter(
            torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size), 
            requires_grad=True
        )
        
        self.n_bins = n_bins
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32).reshape(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32).reshape(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
        self.linear_layer = nn.Linear(lin_dim + n_bins, prompt_len)
        self.conv3x3 = nn.Conv2d(
            prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.num_expert = num_expert
        
    def extract_hog_features(self, x):
        """Extract HOG histogram features"""
        B, C, H, W = x.shape
        
        if C == 3:
            x_gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        else:
            x_gray = x[:, 0:1]
        
        gx = F.conv2d(x_gray, self.sobel_x, padding=1)
        gy = F.conv2d(x_gray, self.sobel_y, padding=1)
        
        magnitude = torch.sqrt(gx**2 + gy**2 + 1e-6)
        orientation = torch.atan2(gy, gx)
        
        orientation_bin = ((orientation + torch.pi) / (2 * torch.pi) * self.n_bins).long() % self.n_bins
        
        hog_hist = torch.zeros(B, self.n_bins, device=x.device)
        for i in range(self.n_bins):
            bin_mask = (orientation_bin == i).float()
            hog_hist[:, i] = (magnitude * bin_mask).mean(dim=[1, 2, 3])
        
        hog_hist = hog_hist / (hog_hist.sum(dim=1, keepdim=True) + 1e-8)
        
        return hog_hist
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        spatial_emb = x.mean(dim=(-2, -1))
        hog_emb = self.extract_hog_features(x)
        combined_emb = torch.cat([spatial_emb, hog_emb], dim=1)
        
        prompt_weights = F.softmax(self.linear_layer(combined_emb), dim=1)
        topk_weights, topk_experts = torch.topk(prompt_weights, self.num_expert)
        
        exp_weights = torch.zeros_like(prompt_weights)
        exp_weights.scatter_(1, topk_experts, prompt_weights.gather(1, topk_experts))
        
        prompt = exp_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * \
                 self.prompt_param.unsqueeze(0).repeat(B, 1, 1, 1, 1, 1).squeeze(1)
        prompt = torch.sum(prompt, dim=1)
        
        prompt = F.interpolate(prompt, (H, W), mode="bilinear")
        prompt = self.conv3x3(prompt)
        
        return prompt

#########################################################################

Conv2d = nn.Conv2d

##########################################################################
## Layer Norm
def to_2d(x):
    return rearrange(x, 'b c h w -> b (h w c)')

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.normalized_shape = normalized_shape
    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5)

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5)


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type="WithBias", dyt_alpha=0.5):
        super(LayerNorm, self).__init__()
        
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        elif LayerNorm_type == 'DynamicTanh':
            try:
                self.body = DynamicTanh(dim, alpha_init_value=dyt_alpha)
            except:
                print(f"[Warning] DynamicTanh failed for dim={dim}, using WithBias")
                self.body = WithBias_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Feed-Forward Network
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, 
                             stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Attention
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = Conv2d(dim*3, dim*3, kernel_size=3, stride=1, 
                                 padding=1, groups=dim*3, bias=bias)
        self.project_out = Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, 
                 LayerNorm_type, num_experts=2, dyt_alpha=0.5):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type, dyt_alpha)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type, dyt_alpha)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


##########################################################################
## Patch Embedding
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x

class SkipPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(SkipPatchEmbed, self).__init__()
        self.proj_1 = Conv2d(in_c, embed_dim, kernel_size=3, stride=2, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj_1(x)
        return x

##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
##---------- HOGformer with DCPT Support -----------------------
class HOGformer(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim=48,
        num_blocks=[4,6,6,8], 
        num_refinement_blocks=4,
        heads=[1,2,4,8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        dyt_alpha=0.5,
        dual_pixel_task=False,
        use_spm=False,
        prompt_len=5,
        num_experts_per_level=[2,2,2,2]
    ):
        super(HOGformer, self).__init__()
        
        self.use_spm = use_spm
        self.LayerNorm_type = LayerNorm_type
        self.dyt_alpha = dyt_alpha
        
        num_experts = num_experts_per_level

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # ===== Encoder =====
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, 
                           bias=bias, LayerNorm_type=LayerNorm_type, num_experts=num_experts[0],
                           dyt_alpha=dyt_alpha) 
            for i in range(num_blocks[0])
        ])
        
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, 
                           bias=bias, LayerNorm_type=LayerNorm_type, num_experts=num_experts[1],
                           dyt_alpha=dyt_alpha) 
            for i in range(num_blocks[1])
        ])
        
        self.down2_3 = Downsample(int(dim*2**1))
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, 
                           bias=bias, LayerNorm_type=LayerNorm_type, num_experts=num_experts[2],
                           dyt_alpha=dyt_alpha) 
            for i in range(num_blocks[2])
        ])

        self.down3_4 = Downsample(int(dim*2**2))
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, 
                           bias=bias, LayerNorm_type=LayerNorm_type, num_experts=num_experts[3],
                           dyt_alpha=dyt_alpha) 
            for i in range(num_blocks[3])
        ])
        
        # ===== SPM Modules (if enabled) =====
        if self.use_spm:
            self.prompt3 = HOGGuidedPromptGenBlock(
                prompt_dim=int(dim*2**2), prompt_len=prompt_len, 
                prompt_size=16, lin_dim=int(dim*2**3), 
                num_expert=min(num_experts[2], prompt_len)
            )
            self.prompt2 = HOGGuidedPromptGenBlock(
                prompt_dim=int(dim*2**1), prompt_len=prompt_len, 
                prompt_size=32, lin_dim=int(dim*2**2), 
                num_expert=min(num_experts[1], prompt_len)
            )
            self.prompt1 = HOGGuidedPromptGenBlock(
                prompt_dim=dim, prompt_len=prompt_len, 
                prompt_size=64, lin_dim=int(dim*2**1), 
                num_expert=min(num_experts[0], prompt_len)
            )
            
            self.reduce_prompt_level3 = Conv2d(int(dim*2**3) + int(dim*2**2), int(dim*2**3), kernel_size=1, bias=bias)
            self.reduce_prompt_level2 = Conv2d(int(dim*2**2) + int(dim*2**1), int(dim*2**2), kernel_size=1, bias=bias)
            self.reduce_prompt_level1 = Conv2d(int(dim*2**1) + dim, int(dim*2**1), kernel_size=1, bias=bias)
        
        # ===== Decoder =====
        self.up4_3 = Upsample(int(dim*2**3))
        self.reduce_chan_level3 = Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, 
                           bias=bias, LayerNorm_type=LayerNorm_type, num_experts=num_experts[2],
                           dyt_alpha=dyt_alpha) 
            for i in range(num_blocks[2])
        ])

        self.up3_2 = Upsample(int(dim*2**2))
        self.reduce_chan_level2 = Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, 
                           bias=bias, LayerNorm_type=LayerNorm_type, num_experts=num_experts[1],
                           dyt_alpha=dyt_alpha) 
            for i in range(num_blocks[1])
        ])
        
        self.up2_1 = Upsample(int(dim*2**1))
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, 
                           bias=bias, LayerNorm_type=LayerNorm_type, num_experts=num_experts[0],
                           dyt_alpha=dyt_alpha) 
            for i in range(num_blocks[0])
        ])
        
        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, 
                           bias=bias, LayerNorm_type=LayerNorm_type, num_experts=num_experts[0],
                           dyt_alpha=dyt_alpha) 
            for i in range(num_refinement_blocks)
        ])

        self.skip_patch_embed1 = SkipPatchEmbed(3, 3)
        self.skip_patch_embed2 = SkipPatchEmbed(3, 3)
        self.skip_patch_embed3 = SkipPatchEmbed(3, 3)
        self.reduce_chan_level_1 = Conv2d(int(dim*2**1)+3, int(dim*2**1), kernel_size=1, bias=bias)
        self.reduce_chan_level_2 = Conv2d(int(dim*2**2)+3, int(dim*2**2), kernel_size=1, bias=bias)
        self.reduce_chan_level_3 = Conv2d(int(dim*2**3)+3, int(dim*2**3), kernel_size=1, bias=bias)

        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
            
        self.output = Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img, hook=False):
        """
        Forward pass with optional hook support for DCPT
        
        Args:
            inp_img: Input degraded image [B, C, H, W]
            hook: If True, intermediate encoder features will be captured
                  via external hook mechanism (for DCPT training)
        
        Returns:
            Restored image [B, C, H, W]
        """
        # Encoder
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        skip_enc_level1 = self.skip_patch_embed1(inp_img)
        inp_enc_level2 = self.reduce_chan_level_1(torch.cat([inp_enc_level2, skip_enc_level1], 1))
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        skip_enc_level2 = self.skip_patch_embed2(skip_enc_level1)
        inp_enc_level3 = self.reduce_chan_level_2(torch.cat([inp_enc_level3, skip_enc_level2], 1))
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        skip_enc_level3 = self.skip_patch_embed3(skip_enc_level2)
        inp_enc_level4 = self.reduce_chan_level_3(torch.cat([inp_enc_level4, skip_enc_level3], 1))
        latent = self.latent(inp_enc_level4)
        
        # ===== Decoder with SPM =====
        if self.use_spm:
            prompt3 = self.prompt3(latent)
            latent = torch.cat([latent, prompt3], 1)
            latent = self.reduce_prompt_level3(latent)
        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        
        if self.use_spm:
            prompt2 = self.prompt2(out_dec_level3)
            out_dec_level3 = torch.cat([out_dec_level3, prompt2], 1)
            out_dec_level3 = self.reduce_prompt_level2(out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        
        if self.use_spm:
            prompt1 = self.prompt1(out_dec_level2)
            out_dec_level2 = torch.cat([out_dec_level2, prompt1], 1)
            out_dec_level2 = self.reduce_prompt_level1(out_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1)
        
        return out_dec_level1 + inp_img