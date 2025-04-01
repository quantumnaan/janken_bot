import torch
import torch.nn as nn
import torch.nn.functional as F

from constatants import *
class HandDetector(nn.Module):
    def __init__(self, num_classes=3):
        super(HandDetector, self).__init__()
        self.num_classes = num_classes
        self.fc1 = nn.Linear(21 * 3, 128)