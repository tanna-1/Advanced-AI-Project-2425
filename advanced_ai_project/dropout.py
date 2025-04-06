import torch.nn as nn

# PyTorch's dropout layers slow down training even when p=0.0


# A custom dropout layer that does not apply dropout when p=0.0
class ConditionalDropout(nn.Dropout):
    def forward(self, input):
        if self.p == 0.0:
            return input
        return super().forward(input)


# A custom dropout layer that does not apply dropout when p=0.0
class ConditionalAlphaDropout(nn.AlphaDropout):
    def forward(self, input):
        if self.p == 0.0:
            return input
        return super().forward(input)
