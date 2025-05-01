from model.model import HubbardWaveFunction
from model.hamiltonian import HubbardHamiltonian
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

n_params = 5
batch = 32
n_sites = 10
input_token_dims = [2, 2]
embed_dim = 32
n_heads = 2
n_layers = 2
dim_feedforward = 64
particle_number = 3
max_len = 100


def optimization_step(
    hamiltonian: HubbardHamiltonian,
    model: HubbardWaveFunction,
    optimizer: torch.optim.Optimizer,
    params: torch.Tensor,
):

    optimizer.zero_grad()

    # Sample from the wave function
    samples = model.sample(
        num_chains=batch,
        chain_length=n_sites,
        params=params,  # type: ignore
    )

    # Estimate < E_loc > based on samples
    e_loc = model.e_loc(
        hamiltonian=hamiltonian,
        params=torch.randn(n_params),  # type: ignore
        sampled_states=samples,  # type: ignore
    )

    e_loc_real = e_loc.real
    e_loc_real.backward()
    optimizer.step()

    return e_loc_real


def main():
    ITERATIONS = 10000
    LR = 1e-2
    T = 1.0
    U = 2.0

    ham = HubbardHamiltonian(t=1.0, U=2.0)

    model = HubbardWaveFunction(
        embed_dim=embed_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        dim_feedforward=dim_feedforward,
        particle_number=particle_number,
        max_len=max_len,
    )

    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    params = torch.tensor(
        [
            T,  # t
            U,  # U
            n_sites,
            particle_number,
            n_params,
        ]
    )

    for i in range(ITERATIONS):
        e_loc_real = optimization_step(
            hamiltonian=ham,
            model=model,
            optimizer=optimizer,
            params=params,
        )

        if i % 10 == 0:
            print(f"Iteration {i}: Loss = {e_loc_real.item()}")

    print("Optimization completed.")


if __name__ == "__main__":
    main()
