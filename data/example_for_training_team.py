import torch

from data import get_batch, prepare_data


# Use a GPU if one is available. Otherwise, use the CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load and prepare the dataset.
train_data, validation_data, tokenizer = prepare_data()

# These values can later be changed by the training person.
batch_size = 32
block_size = 64

# Create one training batch.
x, y = get_batch(
    data=train_data,
    batch_size=batch_size,
    block_size=block_size
)

# Move the batch to the selected device.
x = x.to(device)
y = y.to(device)

print("Device:", device)
print("Vocabulary size:", tokenizer.vocab_size)
print("Training tokens:", len(train_data))
print("Validation tokens:", len(validation_data))
print("Input shape:", x.shape)
print("Target shape:", y.shape)
print("Input data type:", x.dtype)
print("Target data type:", y.dtype)