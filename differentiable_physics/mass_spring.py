import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt
import os


class MassSpringSystem(nn.Module):
    def __init__(self, num_particles, springs, mass=1.0, dt=0.01, gravity=9.81, device="cpu"):
        super().__init__()
        self.device = device
        self.mass = mass
        self.springs = springs
        self.dt = dt
        self.gravity = gravity

        # Particle 0 is fixed at the origin
        self.initial_position_0 = torch.tensor([0.0, 0.0], device=device)

        # Remaining particles are trainable
        self.initial_positions_rest = nn.Parameter(torch.randn(num_particles - 1, 2, device=device))

        # Velocities
        self.velocities = torch.zeros(num_particles, 2, device=device)

    def forward(self, steps):
        positions = torch.cat([self.initial_position_0.unsqueeze(0), self.initial_positions_rest], dim=0)
        velocities = self.velocities

        for _ in range(steps):
            forces = torch.zeros_like(positions)

            # Compute spring forces
            for (i, j, rest_length, stiffness) in self.springs:
                xi, xj = positions[i], positions[j]
                dir_vec = xj - xi
                dist = dir_vec.norm()
                force = stiffness * (dist - rest_length) * dir_vec / (dist + 1e-6)
                forces[i] += force
                forces[j] -= force

            # Apply gravity
            forces[:, 1] -= self.gravity * self.mass

            # Semi-implicit Euler integration
            acceleration = forces / self.mass
            velocities = velocities + acceleration * self.dt
            positions = positions + velocities * self.dt

            # Fix particle 0 at origin
            positions[0] = self.initial_position_0
            velocities[0] = torch.tensor([0.0, 0.0], device=positions.device)

        return positions


def visualize_positions(initial, final, target, save_path="mass_spring_viz.png"):
    plt.figure(figsize=(6, 4))
    plt.scatter(initial[:, 0], initial[:, 1], c='blue', label='Initial', marker='x')
    plt.scatter(final[:, 0], final[:, 1], c='green', label='Final', marker='o')
    plt.scatter(target[:, 0], target[:, 1], c='red', label='Target', marker='*')
    plt.title("Mass-Spring System Positions")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved visualization to {os.path.abspath(save_path)}")
    plt.close()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    system = MassSpringSystem(
        num_particles=args.num_particles,
        springs=[(0, 1, 1.0, args.stiffness)],
        mass=args.mass,
        dt=args.dt,
        gravity=args.gravity,
        device=device,
    )

    optimizer = optim.Adam(system.parameters(), lr=args.lr)
    target_positions = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0]], device=device
    )

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        final_positions = system(args.steps)
        loss = (final_positions - target_positions).pow(2).mean()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % args.log_interval == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss.item():.6f}")

    # Visualization
    initial_positions = torch.cat([system.initial_position_0.unsqueeze(0), system.initial_positions_rest.detach()], dim=0).cpu().numpy()
    visualize_positions(initial_positions, final_positions.detach().cpu().numpy(), target_positions.cpu().numpy())

    print("\nTraining completed.")
    print(f"Final positions:\n{final_positions.detach().cpu().numpy()}")
    print(f"Target positions:\n{target_positions.cpu().numpy()}")


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    system = MassSpringSystem(
        num_particles=args.num_particles,
        springs=[(0, 1, 1.0, args.stiffness)],
        mass=args.mass,
        dt=args.dt,
        gravity=args.gravity,
        device=device,
    )

    with torch.no_grad():
        final_positions = system(args.steps)
        print(f"Final positions after {args.steps} steps:\n{final_positions.cpu().numpy()}")


def parse_args():
    parser = argparse.ArgumentParser(description="Differentiable Physics: Mass-Spring System")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--steps", type=int, default=50, help="Number of simulation steps per forward pass")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step for integration")
    parser.add_argument("--mass", type=float, default=1.0, help="Mass of each particle")
    parser.add_argument("--stiffness", type=float, default=10.0, help="Spring stiffness constant")
    parser.add_argument("--num_particles", type=int, default=2, help="Number of particles in the system")
    parser.add_argument("--mode", choices=["train", "eval"], default="train", help="Mode: train or eval")
    parser.add_argument("--log_interval", type=int, default=100, help="Print loss every n epochs")
    parser.add_argument("--gravity", type=float, default=9.81, help="Gravity strength")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        evaluate(args)


if __name__ == "__main__":
    main()
