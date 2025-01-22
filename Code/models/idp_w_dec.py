# from afgsa import *
from .afgsa import *

EPSILON =  0.01
TILE_SIZE = 4


class IDP_DEC_WNet(nn.Module):
    def __init__(self, 
        in_ch,
        aux_in_ch,
        base_ch,
        num_sa=4,
        block_size=8,
        halo_size=3,
        num_heads=4,
        num_gcp=2
    ):
        super(IDP_DEC_WNet, self).__init__()
        
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
        
        self.IdempotentWeight = IdempotentWeight(base_ch)
        
        self.decoder = nn.Sequential(
            conv_block(
                base_ch, base_ch,
                kernel_size=3,
                padding_mode="reflect",
                padding=1,
                act_type="relu"
            ),
            conv_block(
                base_ch, 3,
                kernel_size=3,
                padding_mode="reflect",
                padding=1,
                act_type=None
            )
        )
        
    def forward(self, noisy, aux):
        out = self.AFGSANet(noisy, aux)
        sqr_attn, attn, out = self.IdempotentWeight(out)
        out = self.decoder(out) + noisy
        return sqr_attn, attn, out
        

class IdempotentWeight(nn.Module):
    def __init__(self, base_ch):
        super(IdempotentWeight, self).__init__()
        self.base_ch = base_ch
        # self.qk = conv_block(
        #     base_ch,
        #     2 * base_ch,
        #     kernel_size=3,
        #     padding=1,
        #     padding_mode="reflect",
        #     act_type="leakyrelu",
        # )
        self.qkv = nn.Linear(
            base_ch, base_ch * 3, bias=True
        )

    def forward(self, x):
        # x = self.qk(x)
        b, c, h, w = (*x.shape,)
        x = x.permute([0, 2, 3, 1])
        qkv = self.qkv(x).permute([0, 3, 1, 2])

        q, k, v = qkv[:, :c, :, :], qkv[:, c : 2 * c, :, :], qkv[:, 2 * c : 3 * c, :, :]
        q = rearrange(
            q, "b c (h t1) (w t2) -> (b h w) (t1 t2) c", t1=TILE_SIZE, t2=TILE_SIZE
        )

        # print(q.shape)

        k = rearrange(
            k, "b c (h t1) (w t2) -> (b h w) (t1 t2) c", t1=TILE_SIZE, t2=TILE_SIZE
        )

        # print(k.shape)

        v = rearrange(
            v, "b c (h t1) (w t2) -> (b h w) (t1 t2) c", t1=TILE_SIZE, t2=TILE_SIZE
        )
        
        # print(v.shape)
        q = torch.tanh(q)
        k = torch.tanh(k)
        sim = torch.einsum("b i d, b j d -> b i j", q, k)
                
        avg = torch.mean(sim, dim=2, keepdim=True)
        attn = (1 / EPSILON) * (sim - avg) + 1/(TILE_SIZE * TILE_SIZE)
                
        sqr_attn = torch.einsum("b i j, b j s -> b i s", attn, attn)

        out = torch.einsum("b i j, b j c -> b i c", attn, v)
        out = rearrange(
            out,
            "(b h w n) (t1 t2) d -> b (n d) (h t1) (w t2)",
            b=b,
            h=h // TILE_SIZE,
            w=w // TILE_SIZE,
            t1=TILE_SIZE,
            t2=TILE_SIZE,
        )

        return sqr_attn, attn, out


# [INFO] Test Codes
# model = IdempotentWeight(256).to("cuda")
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
