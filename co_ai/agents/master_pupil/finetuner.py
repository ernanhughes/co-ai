# co_ai/agents/master_pupil/student_finetuner.py

import torch
import torch.nn as nn
import torch.optim as optim

class PupilModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class PupilFineTuner:
    def __init__(self, input_dim=512, output_dim=768, lr=1e-4):
        self.model = PupilModel(input_dim, output_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def train_step(self, student_input, teacher_output):
        self.model.train()
        pred = self.model(student_input)
        loss = self.loss_fn(pred, teacher_output)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
