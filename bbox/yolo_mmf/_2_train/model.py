import torch
import torch.nn as nn
import torch.nn.functional as F

################ Sample Original Linear layer (with matmul) ################
class OriginalLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return x @ self.weight.t() + self.bias  # classic matmul

################ Sample MatMulFree layer ################
class MatMulFreeLinear(nn.Module):
    def __init__(self, in_features, out_features, scale=1.0):
        super().__init__()
        # Ternary weights: -1, 0, +1 [C_out, C_in]
        self.weight = nn.Parameter(torch.randint(-1, 2, (out_features, in_features)).float())

        # Bias [C_out]
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Learnable scaling factor (very important!) [scalar]
        self.scale = nn.Parameter(torch.tensor(scale)) 

    def forward(self, x):
        # x: [B, C_in], weight: [C_out, C_in], out: [B, C_out]
        B, I = x.shape
        O = self.weight.shape[0]

        # Instead of x @ W.t() we do element-wise addition/subtraction
        
        # Create masks for +1 and -1 weights: [1 where +1, 0 elsewhere] and [1 where -1, 0 elsewhere]
        # [C_out, C_in]

        # Positive indices: where weight == +1
        pos_mask = self.weight == 1                  # [O, I] bool
        pos_indices = pos_mask.nonzero(as_tuple=True)  # (row, col) tuples

        # Negative indices: where weight == -1
        neg_mask = self.weight == -1                 # [O, I] bool
        neg_indices = neg_mask.nonzero(as_tuple=True) # (row, col) tuples

        # Sum positive contributions (pure addition) [B, O]
        out_pos = torch.zeros(B, O, device=x.device)
        if pos_indices[0].numel() > 0:
            out_pos.index_add_(1, pos_indices[0], x[:, pos_indices[1]])

        # Sum negative contributions (pure subtraction) [B, O]
        out_neg = torch.zeros(B, O, device=x.device)
        if neg_indices[0].numel() > 0:
            out_neg.index_add_(1, neg_indices[0], x[:, neg_indices[1]])

        # Final: scale * (positive_sum - negative_sum) + bias
        # [B, O] = scalar * ([B, O] - [B, O]) + [B, O]
        out = self.scale * (out_pos - out_neg) + self.bias[None, :]

        return out

################ Sample Original Conv2d layer (with matmul) ################
class OriginalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        return self.conv(x)

################ Sample MatMulFree Conv2d layer ################
class MatMulFreeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, scale_init=1.0):
        super().__init__()
        # Ternary weights: -1, 0, +1
        weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.randint(-1, 2, weight_shape).float())
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.scale = nn.Parameter(torch.tensor(scale_init)) 

        self.stride = stride
        self.padding = padding

    def forward(self, x):
        B, C_in, H, W = x.shape
        C_out = self.weight.shape[0]
        kH, kW = self.weight.shape[2], self.weight.shape[3]

        # Unfold input to [B, C_in * kH * kW, H_out * W_out]
        unfolded = F.unfold(x, kernel_size=(kH, kW), stride=self.stride, padding=self.padding)
        # unfolded: [B, C_in*kH*kW, H_out*W_out]

        H_out = (H + 2*self.padding - kH) // self.stride + 1
        W_out = (W + 2*self.padding - kW) // self.stride + 1
        N = H_out * W_out

        pos_indices = (self.weight == 1).nonzero(as_tuple=True) 
        neg_indices = (self.weight == -1).nonzero(as_tuple=True)

        # Positive contributions (pure sum)
        out_pos = torch.zeros(B, C_out, N, device=x.device)
        if pos_indices[0].numel() > 0:
            out_pos.index_add_(1, pos_indices[0], unfolded[:, pos_indices[1], :])

        # Negative contributions
        out_neg = torch.zeros(B, C_out, N, device=x.device)
        if neg_indices[0].numel() > 0:
            out_neg.index_add_(1, neg_indices[0], unfolded[:, neg_indices[1], :])

        # Final: scale * (pos - neg) + bias
        # [B, C_out, N] = scalar * ([B, C_out, N] - [B, C_out, N]) + [C_out]
        out = self.scale * (out_pos - out_neg) + self.bias.view(1, -1, 1)
        out = out.view(B, C_out, H_out, W_out)  # [B, C_out, H_out, W_out]

        return out


################ Tests ################

print("Testing MatMulFreeLinear vs OriginalLinear...")
x = torch.randn(4, 64)          # batch 4, dim 64
lin_normal = OriginalLinear(64, 128)
lin_mmf = MatMulFreeLinear(64, 128)

out_normal = lin_normal(x)
out_mmf = lin_mmf(x)

print("Normal output shape:", out_normal.shape)
print("MMF output shape:", out_mmf.shape)
print("\n") 

print("Testing MatMulFreeConv2d vs OriginalConv...")
x = torch.randn(2, 3, 32, 32)  # batch 2, 3 channels, 32×32
conv_normal = OriginalConv(3, 64, 3, padding=1)
conv_mmf = MatMulFreeConv2d(3, 64, 3, padding=1)

out_normal = conv_normal(x)
out_mmf = conv_mmf(x)

print("Normal output shape:", out_normal.shape)   # [2, 64, 32, 32]
print("MMF output shape:", out_mmf.shape)        # same [2, 64, 32, 32]