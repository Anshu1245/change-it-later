import math
import numpy as np
import matplotlib.pyplot as plt

# Set parameters
step_size = 0.01
steps = 20000
m = 5
c = -5
num_samples = 5000
sigmoid_temp = 1
num_seeds = 50  # Number of seeds to run

# Sigmoid function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x * sigmoid_temp))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Initialize lists to store losses across seeds
all_full_losses = []
all_attn_losses = []

# Loop over seeds
for seed in range(num_seeds):
    np.random.seed(seed)
    x = np.random.randn(num_samples)
    d = np.random.uniform(size=num_samples)

    stdv = 1. / math.sqrt(2)
    initial = np.random.uniform(-stdv, stdv, 3)

    # Full model
    w0, w1, w2 = initial
    full_loss = []

    for step in range(steps):
        w0_grad = w0 + w2 / 2 - c
        w1_grad = w1 - m
        w2_grad = w0 / 2 + w2 / 3 - c / 2

        w0 -= step_size * w0_grad
        w1 -= step_size * w1_grad
        w2 -= step_size * w2_grad

        full_loss.append(np.mean(np.square(m * x + c - (w0 + w1 * x + w2 * d))))

    # Attention model
    w0, w1, w2 = initial
    a1, a2 = 0, 0
    attn_loss = []

    for step in range(steps):
        w0_grad = w0 + w2 * sigmoid(a2) / 2 - c
        w1_grad = w1 * (sigmoid(a1) ** 2) - m * sigmoid(a1)
        w2_grad = w0 * sigmoid(a2) / 2 + w2 * ((sigmoid(a2)) ** 2) / 3 - c * sigmoid(a2) / 2
        a1_grad = (w1 ** 2) * sigmoid(a1) * sigmoid_derivative(a1) - m * w1 * sigmoid_derivative(a1)
        a2_grad = w0 * w2 * sigmoid_derivative(a2) / 2 + (w2 ** 2) * sigmoid(a2) * sigmoid_derivative(a2) / 3 - c * w2 * sigmoid_derivative(a2) / 2

        w0 -= step_size * w0_grad
        w1 -= step_size * w1_grad
        w2 -= step_size * w2_grad
        a1 -= step_size * a1_grad
        a2 -= step_size * a2_grad

        attn_loss.append(np.mean(np.square(m * x + c - (w0 + w1 * sigmoid(a1) * x + w2 * sigmoid(a2) * d))))

    # Store losses for this seed
    all_full_losses.append(full_loss)
    all_attn_losses.append(attn_loss)

    print(sigmoid(a1), sigmoid(a2), w1*sigmoid(a1), w2*sigmoid(a2))

# Compute average losses across seeds
avg_full_loss = np.mean(all_full_losses, axis=0)
avg_attn_loss = np.mean(all_attn_losses, axis=0)

# Plot average losses
plt.figure(figsize=(10, 6))
plt.ylim(0, 0.5)
plt.xlim(0, 7000)
plt.plot(avg_full_loss, label="Average Full Loss", color='b')
plt.plot(avg_attn_loss, label="Average Attention Loss", color='r')
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Average Loss Curves Across Seeds")
plt.legend()
plt.savefig('./average_loss_curves.pdf', bbox_inches='tight')
plt.show()