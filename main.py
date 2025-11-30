from collections import Counter, defaultdict

def train_bpe(text, vocab_size=1000):

    word_freq = Counter(text)
    vocab = set()
    word_splits = {}

    for word in word_freq:
        chars = list(word) + ['</w>']
        word_splits[word] = chars
        vocab.update(chars)

    while len(vocab) < vocab_size:

        pair_freq = defaultdict(int)
        for word, freq in word_freq.items():
            chars = word_splits[word]
            for i in range(len(chars) - 1):
                pair = (chars[i], chars[i+1])
                pair_freq[pair] += freq

        if not pair_freq:
            break

        best_pair = max(pair_freq, key=pair_freq.get)
        new_token = ''.join(best_pair)

        for word in word_freq:
            chars = word_splits[word]
            i = 0
            new_chars = []
            while i < len(chars):
                if i < len(chars) - 1 and (chars[i], chars[i+1]) == best_pair:
                    new_chars.append(new_token)
                    i += 2
                else:
                    new_chars.append(chars[i])  
                    i += 1

            word_splits[word] = new_chars

        vocab.add(new_token)

    return vocab


def apply_bpe(word, vocab):

    if not word:
        return []

    chars = list(word) + ['</w>']

    while True:
        pairs = [(chars[i], chars[i+1]) for i in range(len(chars)-1)]
        mergeable = [''.join(pair) for pair in pairs if ''.join(pair) in vocab]

        if not mergeable:
            break

        best_merge = mergeable[0]
        new_chars = []
        i = 0

        while i < len(chars):
            if i < len(chars) - 1 and ''.join(chars[i:i+2]) == best_merge:
                new_chars.append(best_merge)
                i += 2
            else:
                new_chars.append(chars[i])  # <-- 這裡也要 append
                i += 1

        chars = new_chars

    return chars


def main():

    text = ["low", "low", "lower", "lowest", "new", "newer"]

    print("original text:", text)

    vocab_size = 10
    vocab = train_bpe(text, vocab_size)
    print("trained vocab:", sorted(vocab))

    test_words = ["low", "lowest", "newest"]
    for word in test_words:
        tokens = apply_bpe(word, vocab)
        print(f"word '{word}' -> tokens result: {tokens}")

if __name__ == "__main__":
    main()
