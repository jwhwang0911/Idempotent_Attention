# 0. Folders
- Code
  - Only for codes
  - wandb   : for weight and bias logs
  - models  : Variants of models (hierarcy models)
- Data
  - h5      : H5 files
  - Train data
  - Test data
- Result
  - {Model Name} : Model.pt for checkpoints
  - Inference    : Tested datas (Scenes / Spps)

# Experiments
1. Idempotent Loss (L1) = $(H^2 - H)$ : Failed
  - Idempotent Matrix $H$'s i th row $_i$: $[1/N, 1/N \cdots , 1/N]$ (Uniform Matrix)
  - Discussion : Idempotent Loss is hard constraint to learn

2. Idmepotent Variance Loss = $(H^2-H) - {\lambda}_{var} \cdot \Sigma_i(\Sigma_j (H_{ij}-\Sigma_t H_{it} )^2)$ : Failed
  - Same Issue with 1.

# Conclusion

Idempotent is not learnable. Idempotent matrix need to have 0 or 1 eigen value. In case that initial is 0 but the optimal is 1 then the changing 0 to 1 will be not continuous. Finally we can conclude that Idempotent space is non-continuous and the learning direction need to jump on the non-continuous space so it fails.