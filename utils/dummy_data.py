import torch
import torch.nn.functional as F
import einops as ein


def create_uniform_params(n_params: int, b: int) -> torch.Tensor:
    """
    Generate uniform random parameters for testing.

    Args:
        n_params (int): Number of parameters.
        b (int): Batch size.

    Returns:
        torch.Tensor: A tensor of shape (n_params, b) with uniform random values.
    """
    test_params = torch.rand(n_params, 1)
    test_params = ein.repeat(test_params, "n_param 1 -> n_param b", b=b)
    return test_params


def create_params(n_params: int) -> torch.Tensor:
    """
    Generate a single set of params for testing.

    Args:
        n_params (int): Number of parameters.
    """

    test_params = torch.rand(n_params)
    return test_params


def create_occupations(s: int, b: int, sp: int, o: int) -> torch.Tensor:
    """
    Generate test occupations as a one-hot encoded tensor.

    Args:
        s (int): Number of tokens.
        b (int): Batch size.
        sp (int): Spin dimension.
        o (int): Occupation dimension.

    Returns:
        torch.Tensor: A tensor of shape (s, b, o, sp) with one-hot encoded occupations.
    """
    max_occ = 50  # Maximum occupation value
    test_occupations = torch.rand(s, b, sp, o)  # Random values for occupations
    test_occupations = (test_occupations * (max_occ + 1)).to(
        torch.int64
    )  # Scale and convert to integers
    test_occupations = torch.argmax(
        test_occupations, dim=-2
    )  # Find the max along the spin dimension
    test_occupations = F.one_hot(
        test_occupations, num_classes=o
    )  # One-hot encode along the occupation dimension
    test_occupations = ein.rearrange(
        test_occupations, "s b sp o -> s b o sp"
    )  # Rearrange dimensions
    return test_occupations.to(dtype=torch.float32)
