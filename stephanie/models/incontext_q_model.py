from streamlit import json
import torch
import torch.nn as nn
import json
import os
from stephanie.scoring.mrq.encoder import TextEncoder

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class ExpectileHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

class PolicyHead(nn.Module):
    def __init__(self, input_dim, action_dim=1):
        super().__init__()
        self.linear = nn.Linear(input_dim, action_dim)

    def forward(self, x):
        return self.linear(x)  # Optionally softmax if needed

class InContextQModel(nn.Module):
    def __init__(self, dim, hdim, action_dim=1, device="cpu"):
        super().__init__()
        print(f"Initializing InContextQModel with dim={dim}, hdim={hdim}, action_dim={action_dim}, device={device}")
        self.device = device
        self.encoder = TextEncoder(dim, hdim).to(device)
        self.q_head = MLP(dim, 1).to(device)
        self.v_head = ExpectileHead(dim).to(device)
        self.pi_head = PolicyHead(dim, action_dim).to(device)

    def forward(self, prompt_emb, output_emb):
        prompt_emb = prompt_emb.to(self.device)
        output_emb = output_emb.to(self.device)

        zsa = self.encoder(prompt_emb, output_emb)  # shape: [batch, hdim]
        q_value = self.q_head(zsa)
        state_value = self.v_head(zsa)
        action_probabilities = self.pi_head(zsa)

        return {
            "q_value": q_value,
            "state_value": state_value,
            "action_probabilities": action_probabilities,
        }

    @classmethod
    def load_from_path(cls, model_path: str, dim_name: str, device="cpu"):
        """
        Load model weights from a directory:
        - {model_path}/{dim_name}_encoder.pt
        - {model_path}/{dim_name}_q.pt
        - {model_path}/{dim_name}_v.pt
        - {model_path}/{dim_name}_pi.pt
        """
        print(f"Loading InContextQModel from {model_path} for dimension {dim_name} on device {device}")
        encoder_path = os.path.join(model_path, f"{dim_name}_encoder.pt")
        q_path = os.path.join(model_path, f"{dim_name}_q.pt")
        v_path = os.path.join(model_path, f"{dim_name}_v.pt")
        pi_path = os.path.join(model_path, f"{dim_name}_pi.pt")

        with open(os.path.join(model_path, f"{dim_name}.meta.json")) as f:
            meta = json.load(f)

        dim = meta.get("dim", 1024)
        hdim = meta.get("hdim", 512)

        model = cls(dim=dim, hdim=hdim, device=device)
        print(f"Model initialized with dim={dim}, hdim={hdim}")
        model.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        model.q_head.load_state_dict(torch.load(q_path, map_location=device))
        model.v_head.load_state_dict(torch.load(v_path, map_location=device))
        model.pi_head.load_state_dict(torch.load(pi_path, map_location=device))
        model.eval()
        return model
