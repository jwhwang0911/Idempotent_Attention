import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

import models
from preprocessing import Image_Post, preprocess_data
from parameters import device, IMAGE_CONFIG
import exr

PERMUTATION = [0, 3, 1, 2]
DEPERMUTE = [1, 2, 0]

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="Idempotent", choices=["Idempotent", "AFGSA"])
parser.add_argument("--modelPath", type=str, default="/workspace/Result/idp_w_5/model_epoch7/G.pt")
args, extras = parser.parse_known_args()

# [INFO] Define Model
model : nn.Module = None
match(args.model):
    case "AFGSA":
        model = models.AFGSANet(in_ch=3, aux_in_ch=7, base_ch=256).to(device)
    case "Idempotent":
        model = models.Idempotent_WNet(in_ch=3, aux_in_ch=7, base_ch=256).to(device)
        
model.load_state_dict(torch.load(args.modelPath))

# [INFO] Inference

def inference(data: dict[str, np.ndarray]) -> np.ndarray:
        # data: dict[str, np.ndarray] = preprocess_data(inputfile, gtfile)
        noisy = data["noisy"]
        aux = data["aux"]
        noisy = (
            torch.tensor(noisy, dtype=torch.float)
            .unsqueeze(dim=0)
            .permute(PERMUTATION)
            .to(device)
        )
        aux = (
            torch.tensor(aux, dtype=torch.float)
            .unsqueeze(dim=0)
            .permute(PERMUTATION)
            .to(device)
        )
        # inference
        with torch.no_grad():
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)

            starter.record()
            with torch.cuda.amp.autocast():
                sqr_attn, attn, output  = model(noisy, aux)
                output = output.squeeze()

            ender.record()

            output = output.permute(DEPERMUTE).cpu().numpy()

        torch.cuda.synchronize()
        infertime = starter.elapsed_time(ender)

        return Image_Post(output), sqr_attn, attn, infertime * 1e-3

def inference_full_image(inputpath : str, gtpath : str):
    data : dict[str, np.ndarray] = preprocess_data(inputpath, gtpath)
    # * print(data["noisy"].shape) = Ry, Rx, C (세로, 가로, rgb)
    Ry, Rx, c = data["noisy"].shape
    assert Rx % 2 == 0, "Resolution of x axis should be even number"
    
    # [INFO] Split and Overlap Image (Memory Issue)
    OVERLAP = 40
    CENTER  = Rx // 2
    REMAIN  = 0 if CENTER % IMAGE_CONFIG["TileSize"] == 0 else IMAGE_CONFIG["TileSize"] - (CENTER % IMAGE_CONFIG["TileSize"])
    
    # * If there is remainder, some tile could be divided
    split1 = CENTER + REMAIN + OVERLAP
    split2 = Rx-split1
    
    data1 = {}
    data2 = {}
    for key, value in data.items():
        data1[key] = value[:, :split1, :]
        data2[key] = value[:, split2:, :]
        
    output1, _, __, inf_time1 = inference(data1)
    output2, _, __, inf_time2  = inference(data2)
    
    CENTER          = Rx // 2
    output          = np.concatenate([output1[:, :CENTER, :], output2[:, split1-CENTER:, :]], axis=1)
    inference_time  = inf_time1 + inf_time2
    
    return output, inference_time

# [INFO] pos_x : ----, pos_y : | => pox_x:pox_x + pathSize, pos_y : pos_y + patchSize
def inference_crop(inputpath : str, gtpath : str, pos_x : int, pos_y : int):
    data : dict[str, np.ndarray] = preprocess_data(inputpath, gtpath)
    Ry, Rx, c = data["noisy"].shape
    # | check the crop position is valid
    check_crop_pos = pos_x >= 0 and pos_y >= 0 and (pos_x + IMAGE_CONFIG["PatchSize"] < Rx) and (pos_y + IMAGE_CONFIG["PatchSize"] < Ry)
    assert check_crop_pos, f"Invalid Position to crop. Resolution of Image ({Rx}, {Ry}) and given position ({pos_x}, {pos_y})"
    
    cropped_data = {}
    for key, value in data.items():
        cropped_data[key] = value[pos_y:pos_y+IMAGE_CONFIG["PatchSize"], pos_x:pos_x+IMAGE_CONFIG["PatchSize"]]

    output, sqr_attn, attn, inf_time = inference(cropped_data)
    TILE_SIZE = IMAGE_CONFIG["TileSize"]
    
    sqr_sqr_attn = torch.einsum("b i j, b j c -> b i c", sqr_attn, sqr_attn)
    
    sqrsqr_attn = rearrange(
        sqr_sqr_attn,
        "(h w n) (t1 t2) d -> (n d) (h t1) (w t2)",
        h= IMAGE_CONFIG["PatchSize"] // IMAGE_CONFIG["TileSize"],
        w= IMAGE_CONFIG["PatchSize"] // IMAGE_CONFIG["TileSize"],
        t1= IMAGE_CONFIG["TileSize"],
        t2= IMAGE_CONFIG["TileSize"]
    )
    
    sqr_attn = rearrange(
        sqr_attn,
        "(h w n) (t1 t2) d -> (n d) (h t1) (w t2)",
        h= IMAGE_CONFIG["PatchSize"] // IMAGE_CONFIG["TileSize"],
        w= IMAGE_CONFIG["PatchSize"] // IMAGE_CONFIG["TileSize"],
        t1= IMAGE_CONFIG["TileSize"],
        t2= IMAGE_CONFIG["TileSize"]
    )
    
    attn = rearrange(
        attn,
        "(h w n) (t1 t2) d -> (n d) (h t1) (w t2)",
        h= IMAGE_CONFIG["PatchSize"] // IMAGE_CONFIG["TileSize"],
        w= IMAGE_CONFIG["PatchSize"] // IMAGE_CONFIG["TileSize"],
        t1= IMAGE_CONFIG["TileSize"],
        t2= IMAGE_CONFIG["TileSize"]
    )
    
    sqrsqr_attn = sqrsqr_attn.permute(DEPERMUTE).cpu().numpy()
    sqr_attn = sqr_attn.permute(DEPERMUTE).cpu().numpy()
    attn = attn.permute(DEPERMUTE).cpu().numpy()
    
    return output, sqr_attn, attn, inf_time, sqrsqr_attn
    
    
def test_all():
    inputpath = "/workspace/Data/inference/input/cbox/cbox_32.exr"
    gtpath = "/workspace/Data/inference/input/cbox/cbox_32.exr"
    
    output, inference_time = inference_full_image(inputpath, gtpath)
    
    exr.write("/workspace/Data/cbox.exr", output)
    
def test_crop():
    inputpath = "/workspace/Data/inference/input/cbox/cbox_32.exr"
    gtpath = "/workspace/Data/inference/input/cbox/cbox_32.exr"    
    writepath = "/workspace/Data/inference.exr"
    attnwritepath = "/workspace/Data/attn.exr"
    sqrwritepath = "/workspace/Data/sqr.exr"
    sqrsqrwritepath = "/workspace/Data/sqrsqr.exr"
    pos_x = 72
    pos_y = 124
    output, sqr_attn, attn, inf_time, sqrsqr_attn = inference_crop(inputpath, gtpath, pos_x, pos_y)
    
    exr.write(writepath, output)
    attn_write = {}
    sqr_write = {}
    sqrsqr_write = {}
    w, h, N = attn.shape
    
    for i in range(N):
        attn_write[f"{i}"] = attn[:, :, i]
        sqr_write[f"{i}"] = sqr_attn[:, :, i]
        sqrsqr_write[f"{i}"] = sqrsqr_attn[:, :, i]
    
    exr.write(attnwritepath, attn_write)
    exr.write(sqrwritepath, sqr_write)
    exr.write(sqrsqrwritepath, sqrsqr_write)
    
    
test_all()