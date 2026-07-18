import torch

from data import get_batch, prepare_data
from tokenizer import ByteTokenizer


def test_tokenizer():
    tokenizer = ByteTokenizer()

    sample_text = "To be, or not to be."
    tokens = tokenizer.encode(sample_text)
    decoded_text = tokenizer.decode(tokens)

    assert decoded_text == sample_text
    assert tokenizer.vocab_size == 256

    print("Tokenizer test passed.")


def test_data_split():
    train_data, validation_data, tokenizer = prepare_data()

    total_tokens = len(train_data) + len(validation_data)

    assert tokenizer.vocab_size == 256
    assert train_data.dtype == torch.long
    assert validation_data.dtype == torch.long
    assert len(train_data) > len(validation_data)
    assert total_tokens > 1_000_000

    print("Data split test passed.")
    print("Training tokens:", len(train_data))
    print("Validation tokens:", len(validation_data))


def test_batch():
    train_data, _, _ = prepare_data()

    batch_size = 4
    block_size = 8

    x, y = get_batch(
        data=train_data,
        batch_size=batch_size,
        block_size=block_size
    )

    assert x.shape == (batch_size, block_size)
    assert y.shape == (batch_size, block_size)
    assert x.dtype == torch.long
    assert y.dtype == torch.long

    # Every target token should equal the following input token.
    assert torch.equal(x[:, 1:], y[:, :-1])

    print("Batch test passed.")
    print("Input shape:", x.shape)
    print("Target shape:", y.shape)


if __name__ == "__main__":
    test_tokenizer()
    test_data_split()
    test_batch()

    print()
    print("All Task 2 tests passed.")