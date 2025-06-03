import torch
from equinox.cost.cost_rev1 import CostRev1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cost_model_1 = CostRev1(beta0=0.0, beta1=1e-2, beta2=0.0, device=device)
