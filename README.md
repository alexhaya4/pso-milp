# PSO-MILP

A Python-based project for solving Mixed-Integer Linear Programming (MILP) problems using Particle Swarm Optimization (PSO) and related metaheuristics.

## Overview

This repository provides an implementation of Particle Swarm Optimization (PSO) tailored for solving MILP problems. It leverages Python as the primary language and includes performance-critical components in Cython, C, and C++ for computational efficiency.

## Features

- **MILP Solver**: Hybrid approach combining PSO with mathematical programming.
- **Modularity**: Easily extendable framework for experimenting with new heuristics and hybridizations.
- **Performance**: Critical routines accelerated using Cython, C, and C++.
- **Customization**: Define your own MILP problems and PSO configurations.

## Installation

1. **Clone the repository**
    ```bash
    git clone https://github.com/alexhaya4/pso-milp.git
    cd pso-milp
    ```

2. **Install dependencies**
    - It is recommended to use a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

    - If Cython/C/C++ extensions are used, ensure you have a working compiler (e.g., `gcc`).

## Usage

1. **Define your MILP problem**
    - See the `examples/` directory for sample MILP models.

2. **Configure PSO parameters**
    - Set parameters such as population size, inertia, cognitive and social coefficients in the configuration file or script.

3. **Run the solver**
    ```bash
    python main.py --config configs/default.yaml
    ```

4. **Analyze results**
    - Results and logs are saved in the `results/` directory.

## Project Structure

```
pso-milp/
│
├── main.py                # Entry point for the solver
├── pso/                   # PSO algorithm implementation
├── milp/                  # MILP problem definition and handling
├── cython_ext/            # (Optional) Cython extensions for speed
├── examples/              # Example MILP problems and PSO configs
├── results/               # Output from runs
├── configs/               # Configuration files (YAML/JSON)
├── requirements.txt       # Python package dependencies
└── README.md
```

## Configuration

- **PSO Parameters**: Set in `configs/` (e.g., swarm size, number of iterations).
- **MILP Model**: Define in `examples/`.
- **Command-line Arguments**: See `python main.py --help` for available options.

## Contributing

Contributions are welcome! Please open issues or submit pull requests for new features, bug fixes, or improvements.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Inspired by research in hybrid metaheuristics for optimization.
- Uses open-source libraries such as NumPy, SciPy, and Cython.

---

For further information, please refer to the source code and in-line documentation.
