import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Ensure consistent plotting with Agg
plt.switch_backend('agg')

class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future=0):
        outputs = []

        h_t, c_t = torch.zeros(input.size(0), 51, dtype=torch.double, device=input.device), torch.zeros(input.size(0), 51, dtype=torch.double, device=input.device)
        h_t2, c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double, device=input.device), torch.zeros(input.size(0), 51, dtype=torch.double, device=input.device)

        for input_t in input.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)

        for _ in range(future):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)

        return torch.cat(outputs, dim=1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=15, help='steps to run')
    opt = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)

    # Set the device for GPU compatibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = torch.load('traindata.pt')
    input, target = torch.from_numpy(data[3:, :-1]).to(device), torch.from_numpy(data[3:, 1:]).to(device)
    test_input, test_target = torch.from_numpy(data[:3, :-1]).to(device), torch.from_numpy(data[:3, 1:]).to(device)

    seq = Sequence().double().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.LBFGS(seq.parameters(), lr=0.5)

    for i in range(opt.steps):
        print('STEP:', i)

        def closure():
            optimizer.zero_grad()
            out = seq(input)
            loss = criterion(out, target)
            print('loss:', loss.item())
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            future = 1000
            pred = seq(test_input, future=future)
            loss = criterion(pred[:, :-future], test_target)
            print('test loss:', loss.item())
            y = pred.cpu().detach().numpy()  # Move the prediction to CPU for numpy operations

            plt.figure(figsize=(30, 10))
            plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
            plt.xlabel('x', fontsize=20)
            plt.ylabel('y', fontsize=20)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)

            colors = ['r', 'g', 'b']
            for idx, color in enumerate(colors):
                plt.plot(np.arange(input.size(1)), y[idx, :input.size(1)], color, linewidth=2.0)
                plt.plot(np.arange(input.size(1), input.size(1) + future), y[idx, input.size(1):], color + ':', linewidth=2.0)

            plt.savefig(f'predict{i}.pdf')
            plt.close()

if __name__ == '__main__':
    main()
