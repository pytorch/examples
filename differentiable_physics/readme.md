# Differentiable Physics: Mass-Spring System

This example demonstrates a simple differentiable mass-spring system using PyTorch.

Particles are connected by springs and evolve under the forces exerted by the springs and gravity.  
The system is fully differentiable, allowing the optimization of particle positions to match a target configuration using gradient-based learning.

---

## Files

- `mass_spring.py` — Implements the mass-spring simulation, training loop, and evaluation.
- `README.md` — Usage instructions and description.


---

## Requirements

- Python 3.8+
- PyTorch
- pip install -r requirements.txt

No external dependencies are required apart from PyTorch.

---

## Usage

First, ensure PyTorch is installed.

#### Train the system

```bash
python mass_spring.py --mode train


##### Visualization

After training, the system's final positions are compared to the target positions. The plot below illustrates this comparison:

![Mass-Spring System Visualization](mass_spring_viz.png)

