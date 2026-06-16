from bend.models.downstream import CNN
import torch.nn as nn


class SimpleCNN(CNN):
    def __init__(self):
        super().__init__(
            input_size=4, 
            output_size=900, 
            hidden_size=900,  
            kernel_size=10,
            output_downsample_window=900,  
            hidden_size_downstream=900   
        )
        self.linear = nn.Identity()
        for head in ("softmax", "softplus", "sigmoid"):
            if hasattr(self, head):
                delattr(self, head)


