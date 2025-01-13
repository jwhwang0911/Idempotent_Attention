import torch

device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_str : str = "cuda" if torch.cuda.is_available() else "cpu"

PATHS : dict = {
    "inDir"     :   "../Data/Train",
    "H5Dir"     :   "../Data/h5",
    "outDir"    :   "../Result"
}

IMAGE_CONFIG : dict = {
    # "NumPatches" : 30,
    "NumPatches" : 200,
    "PatchSize" : 128,
    "TileSize" : 4,
    "SCENE_NAMES" : ["white-room", "coffee"]
}

TRAIN_ARGS : dict = {
    "Seed" : 356,
    "NumItersForEval" : 10,
    "BatchSize" : 4,
    "MaxSteps" : 20,
    
    "LearningRate" : 0.0001,
    "lrMilestone" : 3,
    "lrG" : 1e-4,
    "lrD" : 1e-4,

    "l1LossW" : 1.0,
    "ganLossW" : 5e-3,
    "gpLossW" : 10.0,
    "idLossW" : 1.0
}