from typing import Optional
import torch
import einops as ein
import matplotlib.pyplot as plt

seen = set()


def get_log_metric(diag: Optional[dict], metric: str):
    if diag is None:
        return None

    run = diag.get("run", None)
    metric_present = "logging_metrics" in diag and metric in diag["logging_metrics"]

    if run is None or not metric_present:
        return None
    else:
        if not metric in seen:
            print(f"Logging metric {metric} to Neptune")
            seen.add(metric)

        return run[metric]


def tensor_to_string(tensor, precision=5):
    """
    Convert a PyTorch tensor to a print-friendly string using scientific notation.

    Args:
        tensor: The PyTorch tensor to convert.
        precision: Number of decimal places for floating-point numbers.

    Returns:
        A string representation of the tensor.
    """

    formatted_string = "[" + ", ".join(f"{x:.{precision}e}" for x in tensor) + "]"
    return formatted_string


def stringify_bin_tens(bin_tensor: torch.Tensor):
    """
    Convert a binary tensor to a string representation.
    """
    as_str = list(bin_tensor)  # list of integers
    as_str = [str(int(i)) for i in as_str]  # force integer conversion
    return "".join(as_str)


def chains_to_strings(samples: torch.Tensor):
    flattened = ein.rearrange(samples, "s b o sp -> (s sp) b o")
    idx = flattened.argmax(dim=-1)  # s b

    strings = []
    for i in range(idx.shape[1]):
        strings.append(stringify_bin_tens(idx[:, i]))

    return strings


def display_heatmap(
    matrix: torch.Tensor,
    y_labels: list[str],
    x_labels: list[str],
    title: str = "Heatmap",
    y_name: str = "Rows",
    x_name: str = "Columns",
):
    """
    Displays a PyTorch matrix as a heatmap grid with custom row and column labels.

    Args:
        matrix (torch.Tensor): The PyTorch matrix to display.
        row_labels (list[str]): Labels for the rows.
        col_labels (list[str]): Labels for the columns.
        title (str): Title of the heatmap.
    """
    if not isinstance(matrix, torch.Tensor):
        raise ValueError("Input must be a PyTorch tensor.")

    if len(y_labels) != matrix.shape[0]:
        raise ValueError(
            "Number of row labels must match the number of rows in the matrix."
        )
    if len(x_labels) != matrix.shape[1]:
        raise ValueError(
            "Number of column labels must match the number of columns in the matrix."
        )

    # Convert the PyTorch tensor to a NumPy array for plotting
    matrix_np = matrix.numpy()

    # Plot the heatmap
    plt.figure(figsize=(12, 12))
    plt.imshow(matrix_np, cmap="coolwarm", interpolation="nearest")
    plt.colorbar(label="Value")
    plt.title(title)

    # Label rows and columns with custom labels
    plt.xticks(
        ticks=range(matrix_np.shape[1]), labels=x_labels, rotation=45, ha="right"
    )
    plt.yticks(ticks=range(matrix_np.shape[0]), labels=y_labels)

    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.tight_layout()  # Adjust layout to prevent label overlap
    plt.show()
