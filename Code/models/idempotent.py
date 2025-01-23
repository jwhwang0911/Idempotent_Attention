# from afgsa import *
from .afgsa import *

EPSILON =  0.01
TILE_SIZE = 4


class Idempotent_WNet(nn.Module):
    def __init__(self, 
        in_ch,
        aux_in_ch,
        base_ch,
        num_sa=5,
        block_size=8,
        halo_size=3,
        num_heads=4,
        num_gcp=2
    ):
        super(Idempotent_WNet, self).__init__()
        
        self.AFGSANet = AFGSANet(
            in_ch,
            aux_in_ch,
            base_ch,
            num_sa,
            block_size,
            halo_size,
            num_heads,
            num_gcp
        )
        
        self.IdempotentWeight = Decoder_Idempotent(base_ch)
        
    def forward(self, noisy, aux):
        out = self.AFGSANet(noisy, aux)
        sqr_attn, attn, out = self.IdempotentWeight(out, noisy)
        return sqr_attn, attn, out
        
# todo : Additional Experiment [Downsampling and infer kernel tensor as {B, K^2, H/4, W/4} ]
class Decoder_Idempotent(nn.Module):
    def __init__(self, base_ch):
        super(Decoder_Idempotent, self).__init__()
        self.base_ch = base_ch
        
        self.conv_1 = conv_block(
            base_ch,
            base_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="reflect",
            act_type="leakyrelu"
        )
        
        self.down_1 = conv_block(
            base_ch,
            base_ch,
            kernel_size=4,
            stride=2,
            padding=1,
            act_type="leakyrelu",
        )
        
        self.conv_2 = conv_block(
            base_ch,
            base_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="reflect",
            act_type="leakyrelu"
        )
        
        self.down_2 = conv_block(
            base_ch,
            TILE_SIZE * TILE_SIZE * TILE_SIZE * TILE_SIZE,
            kernel_size=4,
            stride=2,
            padding=1,
            act_type="leakyrelu",
        )
        
        self.donwsample = nn.Sequential(
            self.conv_1,
            self.down_1,
            self.conv_2,
            self.down_2,
        )

    def forward(self, x, noisy):
        b, c, h, w = x.size()
        
        weight = self.donwsample(x)
        
        weight = rearrange(
            weight, "b (k1 k2) h w -> (b h w) k1 k2", k1=TILE_SIZE * TILE_SIZE, k2=TILE_SIZE * TILE_SIZE
        )

        v = rearrange(
            noisy, "b c (h t1) (w t2) -> (b h w) (t1 t2) c", t1=TILE_SIZE, t2=TILE_SIZE
        )
        
        weight = torch.tanh(weight)
        avg = torch.mean(weight, dim=2, keepdim=True)
        weight = (1/ EPSILON) * (weight - avg) + 1/(TILE_SIZE * TILE_SIZE)
        
        weight_sqr = torch.einsum("b i j, b j s -> b i s", weight, weight)

        out = torch.einsum("b i j, b j c -> b i c", weight, v)
        
        # # print(v.shape)
        # sim = torch.einsum("b i d, b j d -> b i j", q, k)
        # sim = torch.tanh(sim)
                
        # avg = torch.mean(sim, dim=2, keepdim=True)
        # attn = (1 / EPSILON) * (sim - avg) + 1/(TILE_SIZE * TILE_SIZE)
                
        # sqr_attn = torch.einsum("b i j, b j s -> b i s", attn, attn)

        # out = torch.einsum("b i j, b j c -> b i c", attn, v)
        out = rearrange(
            out,
            "(b h w n) (t1 t2) d -> b (n d) (h t1) (w t2)",
            b=b,
            h=h // TILE_SIZE,
            w=w // TILE_SIZE,
            t1=TILE_SIZE,
            t2=TILE_SIZE,
        )

        return weight_sqr, weight, out

# class Normalize(nn.Module):
#     def __init__(self) -> None:
#         super(Normalize, self).__init__()
        
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
        

# # [INFO] Test Codes
# model = Decoder_Idempotent(256).to("cuda")
# B, C, H, W = 1, 256, 80, 80
# data = torch.rand([B, C, H, W]).to("cuda")

# B, C, H, W = 1, 3, 80, 80

# noisy = torch.rand([B, C, H, W]).to("cuda")
# sqr_attn, attn, out = model(data, noisy)
# print(attn.shape, out.shape)
# print(torch.max(attn), torch.min(attn))

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)


# print(count_parameters(model))