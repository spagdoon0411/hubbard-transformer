import torch
import ipdb


def debug_upstream_params(value, model):
    # This function takes the value that was passed to it, embedded in a computation graph,
    # and lists all of the model parameters under its influence. Collects upstream
    # parameter information and sets a debugger breakpoint before returning.

    if model is not None:
        named_params = list(model.named_parameters())

        names = [n for (n, _) in named_params]
        params = [p for (_, p) in named_params]

        grad = torch.autograd.grad(
            outputs=value,
            inputs=params,
            grad_outputs=torch.ones_like(value),
            allow_unused=True,
            retain_graph=True,
        )

        names_and_grad = list(zip(names, grad))

        not_involved = [
            name
            for (name, p), (name2, g) in zip(named_params, names_and_grad)
            if g is None
        ]

        ret = {
            "grads": names_and_grad,
            "not_involved": not_involved,
        }

        ipdb.set_trace()

        return ret
