from pathlib import Path

import torch

from tokenizer import ByteTokenizer


DATA_FOLDER = Path(__file__).resolve().parent
DATA_FILE = DATA_FOLDER / "input.txt"


def load_text():
    """Load the Tiny Shakespeare text file."""
    return DATA_FILE.read_text(encoding="utf-8")


def prepare_data(train_fraction=0.9):
    """Tokenize the text and split it into training and validation data."""
    text = load_text()

    tokenizer = ByteTokenizer()
    token_list = tokenizer.encode(text)

    all_data = torch.tensor(token_list, dtype=torch.long)

    split_index = int(len(all_data) * train_fraction)

    train_data = all_data[:split_index]
    validation_data = all_data[split_index:]

    return train_data, validation_data, tokenizer


def get_batch(data, batch_size, block_size):
    """Create input sequences and next-token target sequences."""

    maximum_start = len(data) - block_size - 1

    start_positions = torch.randint(
        low=0,
        high=maximum_start,
        size=(batch_size,)
    )

    input_sequences = []
    target_sequences = []

    for start in start_positions:
        start = start.item()

        input_sequence = data[start:start + block_size]
        target_sequence = data[start + 1:start + block_size + 1]

        input_sequences.append(input_sequence)
        target_sequences.append(target_sequence)

    x = torch.stack(input_sequences)
    y = torch.stack(target_sequences)

    return x, y


if __name__ == "__main__":
    train_data, validation_data, tokenizer = prepare_data()

    x, y = get_batch(
        data=train_data,
        batch_size=4,
        block_size=8
    )

    print("Batch created successfully.")
    print("Input shape:", x.shape)
    print("Target shape:", y.shape)

    print()
    print("First input sequence:")
    print(x[0])

    print()
    print("First target sequence:")
    print(y[0])

    print()
    print("Decoded input:")
    print(repr(tokenizer.decode(x[0].tolist())))

    print()
    print("Decoded target:")
    print(repr(tokenizer.decode(y[0].tolist())))