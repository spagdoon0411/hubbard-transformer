from typing import Optional

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
