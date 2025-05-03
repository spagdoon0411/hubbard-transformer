from model.model import HubbardWaveFunction
from model.hamiltonian import HubbardHamiltonian
import torch
from typing import Any
import os

# Fixed for this Hubbard model
N_PARAMS = 5

def optimization_step(
    hamiltonian: HubbardHamiltonian,
    model: HubbardWaveFunction,
    optimizer: torch.optim.Optimizer,
    params: torch.Tensor,
    batch_size: int, 
    n_sites: int,
) -> dict:

    optimizer.zero_grad()

    # Sample from the wave function
    samples = model.sample(
        num_chains=batch_size,
        chain_length=n_sites,
        params=params,  # type: ignore
    )

    # Estimate < E_loc > based on samples
    e_loc = model.e_loc(
        hamiltonian=hamiltonian,
        params=torch.randn(N_PARAMS),  # type: ignore
        sampled_states=samples,  # type: ignore
    )

    e_loc_real, e_loc_imag = e_loc.real, e_loc.imag
    e_loc_real.backward()
    optimizer.step()

    return e_loc_real, e_loc_imag


def run_optimization(run: Any, run_params: dict, device: torch.device, log_dir: str) -> dict:
    ham = HubbardHamiltonian(t=run_params["t"], U=run_params["U"])

    model = HubbardWaveFunction(
        embed_dim=run_params["embed_dim"],
        n_heads=run_params["n_heads"],
        n_layers=run_params["n_layers"],
        dim_feedforward=run_params["dim_feedforward"],
        particle_number=run_params["particle_number"],
        max_len=run_params["max_len"],
    )

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

    for i in range(run_params["epochs"]):
        e_loc_real, e_loc_imag = optimization_step(
            batch_size=run_params["batch_size"],
            n_sites=run_params["n_sites"],
            hamiltonian=ham,
            model=model,
            optimizer=optimizer,
            params=params,
        )

        run["loss/epoch/e_loc_real"].log(e_loc_real.item())
        run["loss/epoch/e_loc_imag"].log(e_loc_imag.item())

        if i % 10 == 0:
            print(f"Iteration {i}: Loss = {e_loc_real.item()}")
            pth = os.path.join(log_dir, f"epoch_{i}")
            torch.save(model.state_dict(), pth)

    print("Optimization completed.")

    return {
        "model":  model,
    }