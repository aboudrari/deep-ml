class CharTokenizer:
    def __init__(self, text: str):
        """
        Build a character-level tokenizer from the input text.
        
        Args:
            text: A string used to build the vocabulary.
        """
        # Your code here
        special = ['<BOS>', '<EOS>']
        chars = sorted(set(text))
        vocab = special + chars
        self.stoi = {ch: i for i, ch in enumerate(vocab)}   # char → index
        self.itos = {i: ch for i, ch in enumerate(vocab)}   # index → char
        self.vocab_size = len(vocab)

    def encode(self, text: str) -> list:
        """
        Encode a string into a list of token indices.
        
        Args:
            text: The string to encode.
        Returns:
            List of integer indices.
        """
        return [self.stoi['<BOS>']] + [self.stoi[ch] for ch in text] + [self.stoi['<EOS>']]
        
    def decode(self, indices: list) -> str:
        """
        Decode a list of token indices back into a string.
        
        Args:
            indices: List of integer indices.
        Returns:
            Decoded string.
        """
        # Your code here
        return "".join([self.itos[i] for i in indices])