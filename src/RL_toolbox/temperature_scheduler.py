import torch


class ReduceOnPlateau:
    def __init__(self, agent, mode='min', factor=0.1, patience=10, threshold=0.0001, min_temp=0):
        self.agent = agent
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_temp = min_temp
        self.previous_return = 0
        self.patience_step = 0

    def step(self, temperature, running_return):
        if torch.abs(running_return - self.previous_return) < self.threshold:
            self.patience_step += 1
            if self.patience_step > self.patience:
                temperature *= temperature * self.factor
                self.patience_step = 0
                self.previous_return = running_return
        else:
            self.previous_return = running_return
        return temperature

