from .afgsa import *

class Vanilla_AFGSANet(nn.Module):
    def __init__(
        self,
        in_ch,
        aux_in_ch,
        base_ch,
        num_sa=5,
        block_size=8,
        halo_size=3,
        num_heads=4,
        num_gcp=2,
    ):
        super(Vanilla_AFGSANet, self).__init__()
        assert num_gcp <= num_sa
        
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
        
        self.decoder = nn.Sequential(
                conv_block(
                    base_ch,
                    base_ch,
                    kernel_size=3,
                    padding=1,
                    padding_mode="reflect",
                    act_type="relu",
                ),
                conv_block(
                    base_ch,
                    base_ch,
                    kernel_size=3,
                    padding=1,
                    padding_mode="reflect",
                    act_type="relu",
                ),
                conv_block(
                    base_ch,
                    3,
                    kernel_size=3,
                    padding=1,
                    padding_mode="zeros",
                    act_type=None,
                ),
            )
        
    def forward(self, noisy, aux):
        out = self.AFGSANet(noisy, aux)
        out = self.decoder(out) + noisy
        return torch.zeros([1]), torch.zeros([1]), out