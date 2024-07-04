import torch
from dtw_soft import (
    soft_dtw_batch_same_size,
    backward_recursion_batch_same_size,
    jacobian_product_sq_euc_batch,
)


class SoftDTWFunction_batch_same_size(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, target, gamma):
        loss, R, delta = soft_dtw_batch_same_size(input, target, gamma)

        # save data for backward
        ctx.save_for_backward(input, target, R, delta)
        ctx.gamma = gamma

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        # get value from forward
        x, y, R, delta = ctx.saved_tensors
        E = backward_recursion_batch_same_size(x, y, R, delta, ctx.gamma)
        q = jacobian_product_sq_euc_batch(x, y, E)
        return q / x.shape[0], None, None


class DTWLoss(torch.nn.Module):
    def __init__(self, gamma=1, reduction="mean"):
        super(DTWLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        # Use self.param in your loss computation
        if self.reduction == "mean":
            loss = torch.mean(
                SoftDTWFunction_batch_same_size.apply(input, target, self.gamma)
            )
        elif self.reduction == "sum":
            loss = torch.sum(
                SoftDTWFunction_batch_same_size.apply(input, target, self.gamma)
            )
        else:
            raise
        return loss
