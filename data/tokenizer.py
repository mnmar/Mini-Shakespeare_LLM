class ByteTokenizer:
    """Converts text into UTF-8 byte tokens and back into text."""

    def __init__(self):
        self.vocab_size = 256

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, tokens):
        return bytes(tokens).decode("utf-8", errors="replace")


if __name__ == "__main__":
    tokenizer = ByteTokenizer()

    sample_text = "To be"
    tokens = tokenizer.encode(sample_text)
    decoded_text = tokenizer.decode(tokens)

    print("Original text:", sample_text)
    print("Tokens:", tokens)
    print("Decoded text:", decoded_text)
    print("Vocabulary size:", tokenizer.vocab_size)

    assert decoded_text == sample_text
    assert tokenizer.vocab_size == 256

    print("Tokenizer test passed.")