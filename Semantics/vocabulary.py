import torch
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance as ndm
from constants import WORD_RELATIONSHIP_WEIGHT, SAME_WORD_VALUE, LEARNING_FACTOR, DECAY_FN, POSITION_AWARE_WEIGHT_FN, WORD_SCALING_FACTOR, RELATIONSHIP_SCALING_FN, SECONDHAND_RELATIONSHIP_WEIGHT, EMBEDDING_SIZE, EMBEDDING_NORMALIZATION, PAD_IDX, SENTENCE_SPLITTER
import logging
from Semantics.POS.POS import POS

LOGGER = logging.getLogger(__name__)


class Vocabulary:
    def __init__(self):
        self.adjacency = torch.tensor([])
        self.vocab = {}
        self.words = set()

    def add_word(self, word):
        LOGGER.debug(f"Adding word {word} to vocabulary")
        if word in self.words:
            return False

        if len(self.words) == 0:
            self.words.add(word)
            self.vocab[word] = 0
            self.adjacency = torch.tensor([[SAME_WORD_VALUE]])
            return True

        # Calculate the values to place here
        value = torch.zeros((1, len(self.words)))
        for learned_word, index in self.vocab.items():
            factor = (ndm(learned_word, word) - 1) * -1
            # value += self.adjacency[index] * factor
            for i, v in enumerate(self.adjacency[index]):
                if i == index:
                    continue
                else:
                    value[0, i] += v * factor * WORD_RELATIONSHIP_WEIGHT
                
        self.adjacency = torch.vstack((self.adjacency, value))
        value = torch.cat((value.T, torch.tensor(SAME_WORD_VALUE).reshape(1, -1)))
        self.adjacency = torch.hstack((self.adjacency, value))
        self.vocab[word] = len(self.words)
        self.words.add(word)

    def parse_sentence(self, sentence):
        LOGGER.debug(f"Parsing sentence {sentence}")
        words = SENTENCE_SPLITTER(sentence.upper())
        for i in words:
            if i not in self.vocab:
                self.add_word(i)

        for main_index, main_word in enumerate(words):
            # Loop through words once            
            for secondary_index, secondary_word in enumerate(words):
                if main_word == secondary_word:
                    # Skip same words
                    continue

                distance = abs(main_index - secondary_index)

                idx1, idx2 = self.vocab[main_word], self.vocab[secondary_word]

                val_to_add = (LEARNING_FACTOR *
                     POSITION_AWARE_WEIGHT_FN(distance) / 
                     (self.adjacency[idx1][idx2] + WORD_SCALING_FACTOR))

                value = RELATIONSHIP_SCALING_FN(
                    self.adjacency[idx1][idx2] +
                    val_to_add
                )

                self.adjacency[idx1][idx2] = value
                self.adjacency[idx2][idx1] = value

                for i, v in enumerate(self.adjacency[idx1]):
                    if i == idx1 or i == idx2:
                        continue
                    # Originally multiplied SECONDHAND_RELATIONSHIP_WEIGHT by np.tanh(v)
                    val_2 = RELATIONSHIP_SCALING_FN(
                        v + (val_to_add * SECONDHAND_RELATIONSHIP_WEIGHT) /
                        (self.adjacency[idx1][i] + WORD_SCALING_FACTOR)
                    )
                    self.adjacency[idx1][i] = val_2
                    self.adjacency[i][idx1] = val_2

            main_adjacency_index = self.vocab[main_word]
            adjacency_row = self.adjacency[main_adjacency_index]
            word_indices = set([self.vocab[w] for w in words])
            for i, v in enumerate(adjacency_row):
                if i == main_adjacency_index or i in word_indices:
                    continue

                self.adjacency[main_adjacency_index][i] = DECAY_FN(v)
                self.adjacency[i][main_adjacency_index] = DECAY_FN(v)

    def calculate_centrality(self):
        import heapq
        if self.adjacency.size == 0:
            return {}
        
        # Central sum, (Central word, Central Index)
        centrals = []
        for word, i in self.vocab.items():
            heapq.heappush(centrals, (torch.sum(self.adjacency[i]), word, i))
        return heapq.nlargest(EMBEDDING_SIZE, centrals, key=lambda x: x[0])

    def get_embedding(self, word: str) -> torch.Tensor:
        word = word.upper()
        if word not in self.vocab:
            raise ValueError(f"Word '{word}' not found in vocabulary.")
        
        central_words = self.calculate_centrality()
        word_index = self.vocab[word]

        central_word_indices = [i[2] for i in central_words]

        embedding = torch.tensor([self.adjacency[word_index][i] for i in central_word_indices])

        embedding = EMBEDDING_NORMALIZATION(embedding)

        if self.vocab[word] in central_word_indices:
            embedding[central_word_indices.index(self.vocab[word])] = 1
        if len(embedding) < EMBEDDING_SIZE:
            # Pad the embedding if it's shorter than EMBEDDING_SIZE
            padded_items = torch.tensor([PAD_IDX] * (EMBEDDING_SIZE - len(embedding)))
            embedding = torch.cat((embedding, padded_items))
        return embedding
    
    def get_embedding_from_output(self, output: torch.Tensor) -> tuple[str, int, torch.Tensor]:
        # Assuming output is a 1D array of size EMBEDDING_SIZE

        output = output.reshape(1, -1)        

        output = output[:,:len(self.vocab)].reshape(len(self.vocab))

        output = EMBEDDING_NORMALIZATION(output)

        central_word_indices = [i[2] for i in self.calculate_centrality()]

        central_words = self.adjacency[central_word_indices]

        best_index = -1
        best_word = None
        best_embedding = None
        max_similarity = -2

        for index, column in enumerate(central_words.T):
            output_norm = output / torch.norm(output)
            column_norm = column / torch.norm(column)
            similarity = torch.dot(output_norm, column_norm)
            if similarity > max_similarity:
                max_similarity = similarity
                best_index = index
                best_word = list(self.vocab.keys())[index]
                best_embedding = self.get_embedding(best_word)
        
        return best_word, best_index, best_embedding
    
    def get_embedding_key(self) -> torch.Tensor:
        key = torch.tensor([i[2] for i in self.calculate_centrality()])
        if len(key) < EMBEDDING_SIZE:
            embedding_pad = torch.tensor([-1] * (EMBEDDING_SIZE - len(key)))
            key = torch.cat((key, embedding_pad))
        return key
    
    def embed_sentence(self, sentence: str) -> torch.Tensor:
        embed = self.get_embedding_key()
        for i in SENTENCE_SPLITTER(sentence):
            embed = torch.vstack((embed, self.get_embedding(i)))
        return embed



def write_adjacency_csv(adj_matrix: torch.Tensor, label_to_index: dict, filename: str):
    import csv
    # **Invert** label:index to index:label for easy lookup
    index_to_label = {v: k for k, v in label_to_index.items()}
    
    # **Sanity check**: ensure the matrix is square and matches label count
    n = adj_matrix.shape[0]
    if adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError("Adjacency matrix must be square.")
    if len(index_to_label) != n:
        raise ValueError("Label dictionary must contain exactly one entry per node.")

    # **Sort** labels by index to align with matrix order
    labels = [index_to_label[i] for i in range(n)]

    # **Write** to CSV
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header row (column labels)
        writer.writerow([""] + labels)
        
        # Write each row with row label
        for i, row in enumerate(adj_matrix):
            writer.writerow([labels[i]] + list(row))

if __name__ == '__main__':
    vocab = Vocabulary()
    with open("Semantics/test_sentence.txt", "r") as file:
        for line in file:
            vocab.parse_sentence(line.strip())

    write_adjacency_csv(vocab.adjacency, vocab.vocab, "Semantics/adjacency_matrix.csv")
    print(vocab.calculate_centrality())
    print(vocab.get_embedding("THE"))
    pass
