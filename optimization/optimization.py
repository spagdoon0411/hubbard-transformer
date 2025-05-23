from model.model import HubbardWaveFunction
from model.hamiltonian import HubbardHamiltonian
from model.weight_init import initialize_weights
from utils.hooks import register_hooks_for_all_modules
import torch
from typing import Any, Optional
import pdb
import os
import neptune
from datetime import datetime
from dotenv import load_dotenv
from utils.ground_states import display_psi

# Fixed for this Hubbard model
N_PARAMS = 5

NEPTUNE_TRACKING = False

load_dotenv(".env")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def optimization_step(
    hamiltonian: HubbardHamiltonian,
    model: HubbardWaveFunction,
    optimizer: torch.optim.Optimizer,
    params: torch.Tensor,
    batch_size: int,
    n_sites: int,
    diag: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    One pass of backprop--so one round of batched autoregressive chain-building,
    one calculation of E_loc.
    """

    optimizer.zero_grad()

    # Sample from the wave function
    samples, log_probs = model.sample(
        num_chains=batch_size,
        chain_length=n_sites,
        params=params,  # type: ignore
        compute_log_prob=True,
    )

    # Estimate < E_loc > based on samples
    e_loc = model.e_loc(
        hamiltonian=hamiltonian,
        params=params,  # type: ignore
        sampled_states=samples,  # type: ignore
    )

    e_loc_real, e_loc_imag = e_loc.real, e_loc.imag

    loss = model.surrogate_loss(
        log_probs=log_probs,
        e_loc_values=e_loc_real,
    )

    loss.backward()

    optimizer.step()

    return loss, e_loc_real.mean(), e_loc_imag.mean()


def run_optimization(
    run_params: dict,
    device: torch.device,
    log_dir: str,
    diag: dict,
) -> dict:
    """
    A round of optimization as a function of run parameters.

    - run is the Neptune run logging object.
    - run_params contains model parameters (e.g., number of layers, device).
    """

    ham = HubbardHamiltonian(t=run_params["t"], U=run_params["U"])

    model = HubbardWaveFunction(
        embed_dim=run_params["embed_dim"],
        n_heads=run_params["n_heads"],
        n_layers=run_params["n_layers"],
        dim_feedforward=run_params["dim_feedforward"],
        particle_number=run_params["particle_number"],
        max_len=run_params["max_len"],
        diag=diag,
    )

    register_hooks_for_all_modules(model, diag.get("run", None))
    initialize_weights(model)

    if run := diag.get("run"):
        run["model/architecture"] = str(model)

    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=run_params["learning_rate"],
    )

    params = torch.tensor(
        [
            run_params["t"],  # N
            run_params["U"],
            run_params["embed_dim"],
            run_params["particle_number"],
            N_PARAMS,
        ]
    )

    try:
        for i in range(run_params["epochs"]):
            loss, e_loc_real, e_loc_imag = optimization_step(
                batch_size=run_params["batch_size"],
                n_sites=run_params["n_sites"],
                hamiltonian=ham,
                model=model,
                optimizer=optimizer,
                params=params,
                diag=diag,
            )

            log_grads(model, diag.get("run"))

            if run := diag.get("run"):
                run["loss/epoch/loss"].log(loss.item())
                run["loss/epoch/e_loc_real"].log(e_loc_real.item())
                run["loss/epoch/e_loc_imag"].log(e_loc_imag.item())

            if i % 10 == 0:
                print(
                    f"Iteration {i}: E_loc = {e_loc_real.item()} Loss = {loss.item()}"
                )
                pth = os.path.join(log_dir, f"epoch_{i}.pt")
                torch.save(model.state_dict(), pth)

        print("Optimization completed.")

    except KeyboardInterrupt:
        print("Optimization interrupted.")

    return {
        "model": model,
        "params": params,
    }


def log_grads(
    model: HubbardWaveFunction,
    run: Optional[neptune.Run],
) -> None:
    """
    Log gradients of the model parameters to Neptune.
    """
    if run is None:
        return

    for name, param in model.named_parameters():
        if param.grad is not None:
            run[f"gradients/mean_abs/{name}"].log(param.grad.abs().mean())
        else:
            run[f"gradients/mean_abs/{name}"].log(0)


def main():

    N_TIMES = 100

    for i in range(N_TIMES):

        # Create weight logging folders
        if not os.path.exists("weights"):
            os.makedirs("weights")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(f"weights/{timestamp}", exist_ok=False)
        weight_dir = f"weights/{timestamp}"

        params = {
            "learning_rate": 1e-3,
            "batch_size": 64,
            "n_sites": 4,
            "embed_dim": 32,
            "n_heads": 1,
            "n_layers": 1,
            "dim_feedforward": 64,
            "particle_number": 4,
            "max_len": 100,
            "t": 1.0,
            "U": 2.0,
            "epochs": 1000,
            "device": str(device),
        }

        print(f"Running optimization with device: {device}")

        # Set up Neptune tracking
        run = None
        if NEPTUNE_TRACKING:
            run = neptune.init_run(
                project="spagdoon0411/hubbard-model",
                api_token=os.environ["NEPTUNE_API_TOKEN"],
                tags=["importance/recents"],
                name=f"grad-explosion-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",  # Custom run name
            )
            run["parameters"] = params

        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Run the optimization loop, with tracking
        res = run_optimization(
            run_params=params,
            device=device,
            log_dir=weight_dir,
            diag={
                "run": run,
                "logging_metrics": ["extra/avg_e_loc_summands"],
                "dump_probs": os.path.join("diag", stamp),
                "dump_samples": os.path.join("diag", stamp),
            },
        )

        if run:
            fig, ax = display_psi(
                wv=res["model"],
                num_sites=params["n_sites"],
                params=res["params"],
            )

            run["psi"].upload(fig)

            run.stop()


if __name__ == "__main__":
    main()
