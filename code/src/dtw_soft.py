# third-party library
import torch


def soft_min(list_a, gamma):
    """Softmin function.

    Args:
        list_a (list): list of values
        gamma (float): gamma parameter

    Returns:
        float: softmin value
    """
    assert gamma >= 0, "gamma must be greater than or equal to 0"

    # transform list_a to numpy array
    list_a = torch.Tensor(list_a)

    if gamma == 0:
        _min = torch.min(list_a)
    else:
        z = -list_a / gamma
        log_sum = torch.max(z) + torch.log(torch.sum(torch.exp(z - max(z))))
        _min = -gamma * log_sum
    return _min


def soft_dtw(
    x: torch.Tensor, y: torch.Tensor, gamma: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Soft Dynamic Time Warping.

    Args:
        x (list): length x feature
        y (list): length x feature
        gamma (float, optional): gamma parameter. Defaults to 1.0.

    Returns:
        float: soft-DTW distance
    """
    # initialize DP matrix
    n = x.shape[0]
    m = y.shape[0]
    R = torch.zeros((n + 1, m + 1))
    R[0, 1:] = float("inf")
    R[1:, 0] = float("inf")
    R[0, 0] = 0.0

    try:
        cost = torch.cdist(x, y, p=2) ** 2
    except:
        print(
            "Carefull : x and y are not D-dimensional > 1 features : added 2 dimensions"
        )
        cost = torch.cdist(x.unsqueeze(1), y.unsqueeze(1), p=2) ** 2

    for j in range(1, m + 1):
        for i in range(1, n + 1):
            # calculate minimum
            _min = soft_min([R[i - 1, j], R[i, j - 1], R[i - 1, j - 1]], gamma)

            # update cell
            R[i, j] = cost[i - 1, j - 1] + _min

    return R[-1, -1], R


def soft_dtw_batch(x: torch.Tensor, y: torch.Tensor, gamma: float = 1) -> torch.Tensor:
    """Soft Dynamic Time Warping.

    Args:
        x (list): Channel x length x feature
        y (list): Channel x length x feature
        gamma (float, optional): gamma parameter. Defaults to 1.0.

    Returns:
        float: soft-DTW distance
    """
    batch = x.shape[0]
    dists = torch.zeros(batch)
    for i in range(batch):
        x_i = x[i]
        y_i = y[i]
        dist, _ = soft_dtw(x_i, y_i, gamma)
        dists[i] = dist

    return dists


def backward_recursion(
    x: torch.Tensor, y: torch.Tensor, gamma: float = 1.0
) -> torch.Tensor:
    """backward recursion of soft-DTW

    Args:
        x (torch.Tensor): length x feature
        y (torch.Tensor): length x feature
        gamma (float, optional): gamma parameter. Defaults to 1.0.

    Returns:
        torch.Tensor: E matrix
    """
    n, m = x.shape[0], y.shape[0]
    # intialization

    # compute delta
    try:
        delta = torch.cdist(x, y, p=2) ** 2
    except:
        print(
            "Carefull : x and y are not D-dimensional > 1 features : added 2 dimensions"
        )
        delta = torch.cdist(x.unsqueeze(1), y.unsqueeze(1), p=2) ** 2
    # delta[:-1, m], delta[n, :-1] = 0.0, 0.0
    delta = torch.cat((delta, torch.zeros((n)).reshape(-1, 1)), dim=1)
    delta = torch.cat((delta, torch.zeros((m + 1)).reshape(1, -1)), dim=0)
    delta[n, m] = 0.0

    # compute E
    E = torch.zeros((n + 2, m + 2))
    E[n + 1, m + 1] = 1.0

    # compute R
    _, R = soft_dtw(x, y, gamma=gamma)
    R = torch.cat((R, -float("inf") * torch.ones((n + 1)).reshape(-1, 1)), dim=1)
    R = torch.cat((R, -float("inf") * torch.ones((m + 2)).reshape(1, -1)), dim=0)
    R[n + 1, m + 1] = R[n, m]

    # backward recursion
    for j in range(m, 0, -1):  # ranges from m to 1
        for i in range(n, 0, -1):  # ranges from n to 1
            a = torch.exp((R[i + 1, j] - R[i, j] - delta[i, j - 1]) / gamma)
            b = torch.exp((R[i, j + 1] - R[i, j] - delta[i - 1, j]) / gamma)
            c = torch.exp((R[i + 1, j + 1] - R[i, j] - delta[i, j]) / gamma)
            E[i, j] = E[i + 1, j] * a + E[i, j + 1] * b + E[i + 1, j + 1] * c

    return E[1:-1, 1:-1]


def jacobian_product_sq_euc_optimized(X, Y, E):
    # Expand X and Y to 3D tensors for broadcasting
    X_expanded = X.unsqueeze(1)  # Shape: [m, 1, d]
    Y_expanded = Y.unsqueeze(0)  # Shape: [1, n, d]

    # Compute the squared differences, shape: [m, n, d]
    diff = X_expanded - Y_expanded

    # Compute the weighted differences, shape: [m, n, d]
    weighted_diff = E.unsqueeze(-1) * diff * 2

    # Sum over the second dimension (n) to get the result, shape: [m, d]
    G = weighted_diff.sum(dim=1)
    return G


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
    n = x.shape[1]
    m = y.shape[1]
    b = x.shape[0]

    R = torch.zeros((b, n + 1, m + 1))
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
    batch = x.shape[0]
    n, m = x.shape[1], y.shape[1]

    # intialization
    delta = torch.cat((delta, torch.zeros((batch, n)).reshape(batch, -1, 1)), dim=2)
    delta = torch.cat((delta, torch.zeros((batch, m + 1)).reshape(batch, 1, -1)), dim=1)
    delta[:, n, m] = 0.0

    # compute E
    E = torch.zeros((batch, n + 2, m + 2))
    E[:, n + 1, m + 1] = 1.0

    # compute R
    # _, R = soft_dtw_batch_same_size(x, y, gamma=gamma)
    R = torch.cat(
        (R, -float("inf") * torch.ones((batch, n + 1)).reshape(batch, -1, 1)), dim=2
    )
    R = torch.cat(
        (R, -float("inf") * torch.ones((batch, m + 2)).reshape(batch, 1, -1)), dim=1
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
