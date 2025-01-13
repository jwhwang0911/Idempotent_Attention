from .afgsa import *

EPSILON =  0.1
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
        

class Decoder_Idempotent(nn.Module):
    def __init__(self, base_ch):
        super(Decoder_Idempotent, self).__init__()
        self.base_ch = base_ch
        self.qk = conv_block(
            base_ch,
            2 * base_ch,
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
            act_type="leakyrelu",
        )

    def forward(self, x, noisy):
        # x = self.qk(x)
        b, c, h, w = (*x.shape,)
        qk = self.qk(x)

        q, k = qk[:, :c, :, :], qk[:, c : 2 * c, :, :]
        q = rearrange(
            q, "b c (h t1) (w t2) -> (b h w) (t1 t2) c", t1=TILE_SIZE, t2=TILE_SIZE
        )
        q *= self.base_ch**-0.5

        # print(q.shape)

        k = rearrange(
            k, "b c (h t1) (w t2) -> (b h w) (t1 t2) c", t1=TILE_SIZE, t2=TILE_SIZE
        )

        # print(k.shape)

        v = rearrange(
            noisy, "b c (h t1) (w t2) -> (b h w) (t1 t2) c", t1=TILE_SIZE, t2=TILE_SIZE
        )
        
        # print(v.shape)

        sim = torch.einsum("b i d, b j d -> b i j", q, k)
        sim = F.tanh(sim)
        
        N_attns, h_weight, w_weight = (*sim.shape, )
        assert h_weight == w_weight, "Attention weight should have same height and width (Hat matrix)"
        
        avg = torch.mean(sim, dim=2, keepdim=True)
        attn = (1 / EPSILON) * (sim - avg) + 1/h_weight
        
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
