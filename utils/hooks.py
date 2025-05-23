import neptune
from typing import Optional
import torch
import ipdb


def create_forward_hook(module_name, neptune_run: Optional[neptune.Run]):
    def log_activation_statistics(module, inputs, outputs):
        try:
            if neptune_run is None:
                return

            # Prevent logging indiv entries
            if type(inputs) is not tuple:
                inputs = (inputs,)

            for i, input in enumerate(inputs):
                if input is not None:
                    input_mean = input.mean().item()
                    neptune_run[f"inputs_outputs/{module_name}/input_{i}_mean"].log(
                        input_mean
                    )

                    if input.numel() > 1:
                        input_std = input.std().item()
                        neptune_run[f"inputs_outputs/{module_name}/input_{i}_std"].log(
                            input_std
                        )

            for i, output in enumerate(outputs):
                if output is not None:
                    output_mean = output.mean().item()
                    neptune_run[f"inputs_outputs/{module_name}/output_{i}_mean"].log(
                        output_mean
                    )

                    if output.numel() > 1:
                        output_std = output.std().item()
                        neptune_run[f"inputs_outputs/{module_name}/output_{i}_std"].log(
                            output_std
                        )

        except Exception as e:
            ipdb.set_trace()

    def forward_hook(module, inputs, outputs):
        with torch.no_grad():
            log_activation_statistics(module, inputs, outputs)

    return forward_hook


def register_hooks_for_all_modules(model, neptune_run):
    for module_name, module in model.named_modules():
        if module_name:
            print("Registering hook for module:", module_name)
            hook = create_forward_hook(module_name=module_name, neptune_run=neptune_run)
            module.register_forward_hook(hook)
