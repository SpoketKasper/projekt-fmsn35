import torch
import time

# Example: 50000 parameters (say 500 x 100 matrix)
input_size = 50000
output_size = 1

model = torch.nn.Linear(input_size, output_size)
input_data = torch.randn(1, input_size)  # batch size 1

# Warm-up (to avoid first-run overhead)
output = model(input_data)
loss = output.sum()
loss.backward()
model.zero_grad()

# Time measurement
start = time.time()

for epoch in range(1_000):
    output = model(input_data)
    loss = output.sum()
    loss.backward()
    model.zero_grad()

#torch.cuda.synchronize() if torch.cuda.is_available() else None
end = time.time()

print(f"Forward + backward pass time: {(end - start)/epoch:.5f} seconds")
