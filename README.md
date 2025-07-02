# 1D Finite Element Heat Diffusion with JAX

This repository demonstrates how to solve the one–dimensional heat equation using the finite element method (FEM) implemented in [JAX](https://github.com/google/jax). The notebooks walk through the assembly of element matrices, application of boundary conditions and time integration for a simple solid bar. A short derivation of the local matrices for a 2D bilinear element is also provided.

## Repository Contents

- **`solid_bar.ipynb`** – Step‑by‑step implementation of 1D heat diffusion. It sets up the mesh, builds global mass and stiffness matrices, applies Dirichlet boundary conditions and performs implicit Euler time stepping in JAX.
- **`2d_solib_bar.ipynb`** – Notes on shape functions and local element matrices for a 2D bilinear quadrilateral element.
- **`requirements.txt`** – Python dependencies used by the notebooks.

## Setup

1. Create a virtual environment (optional but recommended).
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch Jupyter to explore and run the notebooks:
   ```bash
   jupyter notebook
   ```

The examples rely only on NumPy, Matplotlib and the CPU version of JAX. GPU/TPU versions of JAX are not required.

## Quick Start

Open `solid_bar.ipynb` in Jupyter and execute the cells sequentially. The notebook covers:

1. **Mesh generation** – discretising the bar into linear elements.
2. **Element matrices** – constructing the local mass and stiffness matrices.
3. **Assembly** – building global matrices and imposing boundary conditions.
4. **Time integration** – advancing the solution with an implicit Euler scheme accelerated by `jax.jit`.
5. **Visualization** – plotting the temperature profile at several time steps.

Running the notebook will produce a graph of temperature diffusion along the bar over time.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
