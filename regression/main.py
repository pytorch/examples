import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# Polynomial degree and target weights/bias
POLY_DEGREE = 4
W_target = torch.randn(POLY_DEGREE, 1) * 5
b_target = torch.randn(1) * 5


def parse_args():
    """Command line arguments"""
    parser = argparse.ArgumentParser(description='Polynomial Regression Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For saving the current model')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    return parser.parse_args()


def make_features(x):
    """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, POLY_DEGREE + 1)], 1)


def f(x):
    """Approximated function. function f(x) = W_target * x + b_target"""
    return x.mm(W_target) + b_target.item()


def poly_desc(W, b):
    """Creates a string description of a polynomial."""
    result = 'y = '
    for i, w in enumerate(W):
        result += '{:+.2f} x^{} '.format(w, i + 1)
    result += '{:+.2f}'.format(b[0])
    return result


def get_batch(batch_size=32):
    """Builds a batch i.e. (x, f(x)) pair."""
    random = torch.randn(batch_size)
    x = make_features(random)
    y = f(x)
    return x, y


class PolyRegressor(torch.nn.Module):
    """Define the model (simple linear regression)"""
    def __init__(self):
        super(PolyRegressor, self).__init__()
        self.fc = torch.nn.Linear(POLY_DEGREE, 1)

    def forward(self, x):
        return self.fc(x)


def train(args, model, device, optimizer, epoch, log_interval=10):
    """Training loop"""
    model.train()
    for batch_idx in range(1, args.epochs + 1):
        # Get a batch of data
        batch_x, batch_y = get_batch(args.batch_size)
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(batch_x)
        loss = F.smooth_l1_loss(output, batch_y)

        # Backward pass
        loss.backward()

        # Apply gradients
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(f'Epoch {epoch} Batch {batch_idx}/{args.epochs} Loss: {loss.item():.6f}')

        # Dry run for a quick check
        if args.dry_run:
            break


def test(model, device):
    """Test function (in this case, we'll use it to print the learned function)"""
    model.eval()
    model.to(device)
    with torch.no_grad():
        print('==> Learned function:')
        print(poly_desc(model.fc.weight.view(-1), model.fc.bias))
        print('==> Actual function:')
        print(poly_desc(W_target.view(-1), b_target))


def main():
    args = parse_args()

    # Set the random seed
    torch.manual_seed(args.seed)

    # Select the device (GPU/CPU)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Initialize the model, optimizer and scheduler
    model = PolyRegressor().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, optimizer, epoch, args.log_interval)
        scheduler.step()

        # Print the learned function after each epoch
        test(model, device)

        if args.save_model:
            torch.save(model.state_dict(), "polynomial_regressor.pt")

    print("Training complete.")
    if args.save_model:
        print("Model saved to polynomial_regressor.pt")


if __name__ == '__main__':
    main()
