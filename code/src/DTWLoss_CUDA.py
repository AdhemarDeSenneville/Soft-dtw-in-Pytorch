import torch

# from dtw_soft import soft_dtw_batch_same_size, backward_recursion_batch_same_size, jacobian_product_sq_euc_batch


def soft_min_batch(list_a, gamma):
    """Softmin function.

    Args:
        list_a (list): list of values
        gamma (float): gamma parameter

    Returns:
        float: softmin value
    """

    assert gamma >= 0, "gamma must be greater than or equal to 0"
    # Assuming list_a is a list of tensors of the same shape
    list_a = torch.stack(list_a)  # Shape: [n, m]

    if gamma == 0:
        _min = torch.min(list_a, dim=0)[0]  # Min along the first dimension
    else:
        z = -list_a / gamma
        max_z = torch.max(z, dim=0, keepdim=True)[0]  # Max along the first dimension
        log_sum = max_z + torch.log(torch.sum(torch.exp(z - max_z), dim=0))
        _min = -gamma * log_sum
    return _min


def soft_dtw_batch_same_size(
    x: torch.Tensor, y: torch.Tensor, gamma: float = 1
) -> torch.Tensor:
    """Soft Dynamic Time Warping.

    Args:
        x (list): batch x length x feature
        y (list): batch x length x feature
        gamma (float, optional): gamma parameter. Defaults to 1.0.

    Returns:
        float: soft-DTW distance
    """

    # initialize DP matrix
    device = x.device
    n = x.shape[1]
    m = y.shape[1]
    b = x.shape[0]

    R = torch.zeros((b, n + 1, m + 1)).to(device)
    R[:, 0, 1:] = float("inf")
    R[:, 1:, 0] = float("inf")
    R[:, 0, 0] = 0.0

    try:
        cost = torch.cdist(x, y, p=2) ** 2
    except:
        print(
            "Carefull : x and y are not D-dimensional > 1 features : added 2 dimensions"
        )
        cost = torch.cdist(x.unsqueeze(1), y.unsqueeze(1), p=2) ** 2

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            R[:, i, j] = cost[:, i - 1, j - 1] + soft_min_batch(
                [R[:, i - 1, j], R[:, i, j - 1], R[:, i - 1, j - 1]], gamma
            )

    return R[:, -1, -1], R, cost


def backward_recursion_batch_same_size(
    x: torch.Tensor, y: torch.Tensor, R, delta, gamma: float = 1.0
) -> torch.Tensor:
    """backward recursion of soft-DTW

    Args:
        x (torch.Tensor): batch x length x feature
        y (torch.Tensor): batch x length x feature
        gamma (float, optional): gamma parameter. Defaults to 1.0.

    Returns:
        torch.Tensor: E batch x matrix
    """
    device = x.device
    batch = x.shape[0]
    n, m = x.shape[1], y.shape[1]

    # intialization
    delta = torch.cat(
        (delta, torch.zeros((batch, n)).reshape(batch, -1, 1).to(device)), dim=2
    )
    delta = torch.cat(
        (delta, torch.zeros((batch, m + 1)).reshape(batch, 1, -1).to(device)), dim=1
    )
    delta[:, n, m] = 0.0

    # compute E
    E = torch.zeros((batch, n + 2, m + 2)).to(device)
    E[:, n + 1, m + 1] = 1.0

    # compute R
    # _, R = soft_dtw_batch_same_size(x, y, gamma=gamma)
    R = torch.cat(
        (
            R,
            -float("inf") * torch.ones((batch, n + 1)).reshape(batch, -1, 1).to(device),
        ),
        dim=2,
    )
    R = torch.cat(
        (
            R,
            -float("inf") * torch.ones((batch, m + 2)).reshape(batch, 1, -1).to(device),
        ),
        dim=1,
    )
    R[:, n + 1, m + 1] = R[:, n, m]

    # backward recursion
    for j in range(m, 0, -1):  # ranges from m to 1
        for i in range(n, 0, -1):  # ranges from n to 1
            a = torch.exp((R[:, i + 1, j] - R[:, i, j] - delta[:, i, j - 1]) / gamma)
            b = torch.exp((R[:, i, j + 1] - R[:, i, j] - delta[:, i - 1, j]) / gamma)
            c = torch.exp((R[:, i + 1, j + 1] - R[:, i, j] - delta[:, i, j]) / gamma)
            E[:, i, j] = (
                E[:, i + 1, j] * a + E[:, i, j + 1] * b + E[:, i + 1, j + 1] * c
            )

    return E[:, 1:-1, 1:-1]


def jacobian_product_sq_euc_batch(X, Y, E):
    # Expand X and Y to 4D tensors for broadcasting, shape: [b, m, 1, d] and [b, 1, n, d]
    X_expanded = X.unsqueeze(2)
    Y_expanded = Y.unsqueeze(1)

    # Compute the squared differences, shape: [b, m, n, d]
    diff = X_expanded - Y_expanded

    # Adjust E for broadcasting, shape: [b, m, n, d]
    E_adjusted = E.unsqueeze(-1)

    # Compute the weighted differences, shape: [b, m, n, d]
    weighted_diff = E_adjusted * diff * 2

    # Sum over the third dimension (n) to get the result, shape: [b, m, d]
    G = weighted_diff.sum(dim=2)
    return G


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
        return q / x.shape[0] / x.shape[1], None, None


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
