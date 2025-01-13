import argparse
import os
from weakref import ref
import wandb
import math
import time
import warnings

warnings.filterwarnings("ignore")

# * Pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler


# * Code
from metric import calculate_rmse
from preprocessing import Image_Post, hdr_to_ldr
from loss import *
import models
from Dataset import Dataset
from HDF5Constructor import Hdf5Constructor
from prefetch_dataloader import DataLoaderX
from util import create_folder
from parameters import *

DATA_RATIO = (0.95, 0.05)
PERMUTATION = [0, 3, 1, 2]
DEPERMUTE = [1, 2, 0]

# [INFO] Parsers

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="Idempotent", choices=["Idempotent", "AFGSA"])
parser.add_argument("--loadModel", type=bool, default=False)
parser.add_argument("--modelPath", type=str, default="../Result/")
args, extras = parser.parse_known_args()

def wandb_init():
    configuration = {
        "learning_rate" : TRAIN_ARGS["LearningRate"],
        "epochs" : TRAIN_ARGS["MaxSteps"],
        "BatchSize" : TRAIN_ARGS["BatchSize"],
        "WindowSize" : IMAGE_CONFIG["TileSize"],
    }
    
    wandb.init(project="PathReused Denoising", config = configuration)
    wandb.run.name = args.model
    wandb.run.save()


# * wandb validation images (to check model per evaluations)
def wandb_image_subset(val_dataloader : DataLoaderX):
    wandb_image = val_dataloader.random_item()
    valid_noisy = Image_Post(wandb_image["noisy"].cpu().numpy())
    valid_gt = Image_Post(wandb_image["gt"].cpu().numpy())
    valid_torch_gt = torch.tensor(
        wandb_image["gt"].permute(PERMUTATION).clone().detach(), dtype=torch.float32
    ).to(device)
    valid_input = (wandb_image["noisy"]).to(device)
    valid_aux = wandb_image["aux"].to(device)

    valid_input = valid_input.permute(PERMUTATION)
    valid_input = torch.tensor(valid_input.clone().detach())

    valid_aux = valid_aux.permute(PERMUTATION)
    valid_aux = torch.tensor(valid_aux.clone().detach())

    valid_noisy = hdr_to_ldr(valid_noisy)
    valid_gt = hdr_to_ldr(valid_gt)
    
    return valid_noisy, valid_gt, valid_input, valid_aux
    

def main():
    create_folder(PATHS["inDir"])
    create_folder(PATHS["H5Dir"])
    
    root_save_path      = os.path.join(PATHS["outDir"], str(args.model))
    train_save_path     = os.path.join(PATHS["H5Dir"], "train.h5")
    val_save_path       = os.path.join(PATHS["H5Dir"], "val.h5")
    exist = True
    for path in [train_save_path, val_save_path]:
        if not os.path.exists(path):
            exist = False
    
    # [INFO] Wandb Initialize 
    wandb_init()
    
    if not exist:
        constructor = Hdf5Constructor(
            PATHS["inDir"],
            PATHS["H5Dir"],
            IMAGE_CONFIG["PatchSize"],
            IMAGE_CONFIG["NumPatches"],
            TRAIN_ARGS["Seed"],
            DATA_RATIO
        )
        constructor.construct_hdf5()
        
    
        
    # [INFO] 1. Torch Config 
    
    torch.manual_seed(TRAIN_ARGS["Seed"])
    torch.cuda.manual_seed(TRAIN_ARGS["Seed"])
    torch.backends.cudnn.deterministic = True
    
    # [INFO] 2. Dataloader 
    
    train_dataset       = Dataset(train_save_path)
    len_train           = len(train_dataset)
    train_dataloader    = DataLoaderX(train_dataset, TRAIN_ARGS["BatchSize"], shuffle=True, num_workers=7, pin_memory=True)
    
    val_dataset         = Dataset(val_save_path)
    len_val             = len(val_dataset)
    val_dataloader      = DataLoaderX(val_dataset, batch_size = 1, shuffle=True, num_workers=7, pin_memory=True)
    
    # * Train 
    train(train_dataloader, len_train, val_dataloader, len_val, root_save_path)
    
def train(
    train_dataloader: DataLoaderX,
    train_num_samples,
    val_dataloader: DataLoaderX,
    val_num_samples,
    root_save_path,
):
    # [INFO] 3. Define Model 
    model : nn.Module = None
    match(args.model):
        case "AFGSA":
            model = models.AFGSANet(in_ch=3, aux_in_ch=7, base_ch=256).to(device)
        case "Idempotent":
            model = models.Idempotent_WNet(in_ch=3, aux_in_ch=7, base_ch=256).to(device)
    
    # * Load Model when the previous train has been stopped
    if args.loadModel:
        model.load_state_dict(torch.load(args.modelPath))
        
    discriminator = models.DiscriminatorVGG128(3, 64).to(device)
    
    # [INFO] 4. Loss & Optimizer Define
    l1_loss     = L1ReconstructionLoss().to(device)
    gan_loss    = GANLoss("wgan").to(device)
    gp_loss     = GradientPenaltyLoss(device).to(device)
    idp_loss    = IdempotentLoss().to(device)

    milestones = [
        i * TRAIN_ARGS["lrMilestone"] - 1 for i in range(1, TRAIN_ARGS["MaxSteps"] // TRAIN_ARGS["lrMilestone"])
    ]
    optimizer_generator = optim.Adam(
        model.parameters(), lr=TRAIN_ARGS["lrG"], betas=(0.9, 0.999), eps=1e-8
    )
    scheduler_generator = lr_scheduler.MultiStepLR(
        optimizer_generator, milestones=milestones, gamma=0.5
    )
    optimizer_discriminator = optim.Adam(
        discriminator.parameters(), lr=TRAIN_ARGS["lrD"], betas=(0.9, 0.999), eps=1e-8
    )
    scheduler_discriminator = lr_scheduler.MultiStepLR(
        optimizer_discriminator, milestones=milestones, gamma=0.5
    )
    
    # [INFO] 5. Informations and variables for Logs
    iteration = 0
    accum_loss = 0.0    
    idp_accum_loss = 0.0
    accumulated_generator_loss = 0
    accumulated_discriminator_loss = 0
    total_iteraions = math.ceil(train_num_samples / TRAIN_ARGS["BatchSize"])
    
    wandb.watch(model, l1_loss, log="all")
    # * numpy, numpy, tensor, tensor
    wandb_noisy, wandb_gt, wandb_input, wandb_aux = wandb_image_subset(val_dataloader)
    
    # [INFO] 6. Start Training
    for epoch in range(TRAIN_ARGS["MaxSteps"]):
        start = time.time()
        for i_batch, batch_sample in enumerate(train_dataloader):
            iteration   += 1
            
            noisy_input     = torch.tensor(batch_sample["noisy"], dtype=torch.float).permute(PERMUTATION).to(device)
            aux_input       = torch.tensor(batch_sample["aux"], dtype=torch.float).permute(PERMUTATION).to(device)
            ref             = torch.tensor(batch_sample["gt"], dtype=torch.float).permute(PERMUTATION).to(device)
            
            # ! AFGSA will return dummy, dummy, output
            sqr_attn, attn, output = model(noisy_input, aux_input)
            
            # | Discriminator Optimization
            optimizer_discriminator.zero_grad()
            pred_d_fake = discriminator(output.detach())
            pred_d_real = discriminator(ref)
            
            try:
                loss_d_fake = gan_loss(pred_d_fake, False)
                loss_d_real = gan_loss(pred_d_real, True)
                loss_gp     = gp_loss(discriminator, ref, output.detach())
            except:
                break
            
            discriminator_loss = (loss_d_fake + loss_d_real) / 2 + TRAIN_ARGS["gpLossW"] * loss_gp
            discriminator_loss.backward()
            optimizer_discriminator.step()
            accumulated_discriminator_loss += discriminator_loss.item() / TRAIN_ARGS["BatchSize"]
            
            # | Generator Optimization
            optimizer_generator.zero_grad()
            pred_g_fake = discriminator(output.detach())
            
            try:
                loss_g_fake = gan_loss(pred_g_fake, True)
                loss_l1     = l1_loss(output, ref)
                loss_idp    = idp_loss(attn, sqr_attn)
            except:
                break
            
            generator_loss = TRAIN_ARGS["ganLossW"] * loss_g_fake + TRAIN_ARGS["l1LossW"] * loss_l1 + TRAIN_ARGS["idLossW"] * loss_idp
            generator_loss.backward()
            optimizer_generator.step()
            accumulated_generator_loss += generator_loss.item() / TRAIN_ARGS["BatchSize"]
            accum_loss += loss_l1
            idp_accum_loss += loss_idp
            
            if i_batch == 0:
                iter_took = time.time() - start
            else:
                iter_took = time.time() - end
            end = time.time()
            
            # | Logging
            print(
                "\r\t-Epoch : %d\t Took %f sec \tIteration : %d/%d \t Iter Took : %f sec\tG Loss : %f \tD Loss : %f, IDP Loss : %f"
                % (epoch+1, end-start, i_batch + 1, total_iteraions, iter_took, generator_loss.item(), discriminator_loss.item(), loss_idp.item()),
                end=""
            )
            
            if(iteration + 1) % TRAIN_ARGS["NumItersForEval"] == 0:
                with torch.no_grad():
                    valid_loss_l1 = 0.0
                    valid_loss_idp = 0.0
                    for i_val, val_sample in enumerate(val_dataloader):
                        val_noisy   = torch.tensor(val_sample["noisy"], dtype=torch.float).permute(PERMUTATION).to(device)
                        val_aux     = torch.tensor(val_sample["aux"], dtype=torch.float).permute(PERMUTATION).to(device)
                        val_ref     = torch.tensor(val_sample["gt"], dtype=torch.float).permute(PERMUTATION).to(device)
            
                        val_sqr_attn, val_attn, val_output = model(val_noisy, val_aux)
                        
                        valid_loss_l1   += l1_loss(val_output, val_ref)
                        valid_loss_idp  += idp_loss(val_attn, val_sqr_attn)
                    
                    valid_loss_l1 = valid_loss_l1 / (i_val + 1)
                    valid_loss_idp = valid_loss_idp / (i_val + 1)
                    
                    wandb_sqr_attn, wandb_attn, wandb_output = model(wandb_input, wandb_aux)
                    wandb_output = wandb_output.permute([0, 2, 3, 1]).cpu().numpy()
                    wandb_output = hdr_to_ldr(Image_Post(wandb_output))

                    accum_loss /= TRAIN_ARGS["NumItersForEval"]
                    idp_accum_loss /= TRAIN_ARGS["NumItersForEval"]
                    
                    wandb_valid_images : list = [
                        wandb.Image(
                            np.concatenate((t_noisy, t_output, t_gt), axis=1),
                            caption=f"Image : Noisy({calculate_rmse(t_noisy, t_gt)}) |\t Predict({calculate_rmse(t_output, t_gt)})\t | Ground Truth\t",
                        )
                        for t_noisy, t_output, t_gt in zip(
                            wandb_noisy, wandb_output, wandb_gt
                        )
                    ]
                    
                    wandb.log(
                        {
                            "Validation Sample" : wandb_valid_images,
                            "valid_loss" : valid_loss_l1,
                            "valid_loss" : valid_loss_idp,
                            "train_loss" : accum_loss.detach().clone(),
                            "idempotent_loss" : idp_accum_loss.detach().clone()
                        },
                        step = int((iteration + 1) / TRAIN_ARGS["NumItersForEval"])
                    )
                    
                    accum_loss = 0.0
                    idp_accum_loss = 0.0
            end = time.time()
        print(
            "\r\t-Epoch: %d \tG loss: %f \tD Loss: %f \tTook: %d seconds"
            % (
                epoch + 1,
                accumulated_generator_loss / (i_batch + 1),
                accumulated_discriminator_loss / (i_batch + 1),
                end - start,
            )
        )
        
        scheduler_discriminator.step()
        scheduler_generator.step()
        accumulated_generator_loss = 0.0
        accumulated_discriminator_loss = 0.0
        
        with torch.no_grad():
            with open(os.path.join(root_save_path, "loss.txt"), "a") as f:
                f.write(f"epoch {epoch}, iteration : {iteration} \t: loss {accum_loss / (i_batch+1)}\n")
            
        with torch.no_grad():
            current_save_path = create_folder(os.path.join(
                            root_save_path,
                            "model_epoch%d" % (epoch),
                        ))

            torch.save(model.state_dict(), os.path.join(current_save_path, "G.pt"))
            torch.save(discriminator.state_dict(), os.path.join(current_save_path, "D.pt"))
                
        print()
    wandb.finish()
            
    

if __name__ == "__main__": 
    main()