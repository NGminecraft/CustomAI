from vocabulary import Vocabulary, write_adjacency_csv

vocab = Vocabulary()
while True:
    sentence = input("Enter a sentence (or 'exit' to quit): ")
    if sentence.lower() == 'exit':
        break
    vocab.parse_sentence(sentence)
    print(f"Current vocabulary size: {len(vocab.words)}")
    print(f"Adjacency matrix shape: {vocab.adjacency.shape}")
    write_adjacency_csv(vocab.adjacency, vocab.vocab, "Semantics/live.csv")