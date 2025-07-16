import logging
from random import choice, choices
import heapq

LOGGER = logging.getLogger(__name__)

class POS_manager:
    def __init__(self):
        self.tags: list[str]
        self._pos_tags: dict[str, POSGraph] = {}
        self._words: dict[str, Word|LockedWord] = {}
        self.initialized: bool = False
        self.distance_from_end = {}
        self.distances_to_path = {}
        self.end_token: str = ""
        self.start_token: str = ""

    def get_tag(self, tag_name: str) -> "POSGraph | None":
        if tag_name in self._pos_tags:
            return self._pos_tags[tag_name]
        else:
            LOGGER.error(f"POS tag '{tag_name}' not found.")
            return 
    
    def add_tag(self, tag_name: str, tag_connections: list[str]):
        if tag_name in self._pos_tags:
            LOGGER.warning(f"POS tag '{tag_name}' already exists. Updating connections.")
        self._pos_tags[tag_name] = POSGraph(self, tag_name, tag_connections)
        self.tags.append(tag_name)
        self.initialized = False

    def establish(self, end_token, start_token):
        """Setup the reverse connections on all the nodes"""
        LOGGER.debug("Creating backwards paths")
        for i in self._pos_tags.values():
            i.apply_backwards()
        """
        self.distance_from_end = self.shortest_path(end_token) # Calculate the shortest path between every node and the end
        self.distances_to_path = {(i[0], i[1][0][-1]): i[1] for i in self.distance_from_end.values()}
        """
        self.end_token = end_token
        self.start_token = start_token
        self.initialized = True

    @DeprecationWarning
    def shortest_path(self, token: str) -> dict[str,tuple[int, list[tuple[str, ...]]]]:
        """Calculates distance from token to every other token backwards"""
        distances: dict[str,tuple[int, list[tuple[str, ...]]]] = {}
        heap: list[tuple[int, tuple[str]]] = []
        # Initial condition
        heapq.heappush(heap, (0, token, ())) # type: ignore
        distance = 0
        while len(heap) != 0:
            distance, path = heapq.heappop(heap)
            pos_id = path[-1]
            # If we have found a shorter path just skip this
            if pos_id in distances and distance > distances[pos_id][0]:
                continue

            # Create a new item, or append the path to track muliple shortest paths
            if pos_id not in distances:
                distances[pos_id] = (distance, [path, ])
            else:
                distances[pos_id][1].append(path)

            # Append the next nodes in the graph
            self._pos_tags[pos_id].djikstras(heap, offset=distance, path=path)

        return distances
    
    def match_sentence(self, sentence: list[str]) -> list[str] | None:
        sentence_length = len(sentence)
        
        if sentence_length < 4: # Start token, interjection, punctuation, end token. Smallest sentence possible
            LOGGER.error(f"Sentence '{sentence}' has to few tokens!")
            return

    
        word_class_sentence: list[Word|LockedWord] = []
        locked_words: list[tuple[int, str]] = []
        # Get word classes for everything
        for i, v in enumerate(sentence[1:-1]):
            new_class = self.get_word_class(v)
            word_class_sentence.append(new_class)
    
            # Extract the locked words
            if new_class:
                locked_words.append((i, str(new_class)))

        # Add the end token as a locked word
        locked_words.append((sentence_length - 1, self.end_token))

        guessed_tokens = [self.start_token, ]
        probabilities = []
        while len(guessed_tokens) < sentence_length:
            current_tag = guessed_tokens[-1]
            target_distance, target_token = locked_words.pop(0)

            heap: list[tuple[int, tuple[str,...]]] = [(0, (current_tag, ))]
            matches = []
            while len(heap) > 0:
                distance, path = heapq.heappop(heap)
                if distance > target_distance:
                    # The smallest item in the heap was too large, so we can just exit
                    break
                elif path[-1] == target_token and distance == target_distance:
                    # WE FOUND ONE!
                    matches.append(path)
                else:
                    self[path[-1]].djikstras(heap, path, distance, False)
            if len(matches) == 0:
                LOGGER.error(f"The sentence '{sentence}' isn't parsing correctly")
                return []
            final_match = choice(matches)
            guessed_tokens.extend(final_match)
            # Set the probability based on how many other valid matches couldv'e worked
            probabilities.extend([1 / len(matches) for _ in final_match])

        # Update the probability of all the words
        for i, v in enumerate(word_class_sentence):
            v.update_tag(guessed_tokens[i], (guessed_tokens[i-1], guessed_tokens[i+1]), probabilities[i])

        return guessed_tokens
            

        
    def get_word_class(self, word):
        """Gets the word class for a word, or creates it if needed"""
        if word in self._words:
            return self._words[word]
        else:
            self._words[word] = Word(word, self.tags, self)
            return self._words[word]

    def __getitem__(self, item: str) -> "POSGraph":
        return self._pos_tags[item]


class POSGraph:
    def __init__(self, manager: POS_manager, value: str, connections: list[str]):
        self.value = value
        
        self.connections: set[str] = set(connections)
        self.backwards: set[str] = set()
        self.max_permutations: int = 0

        self.manager = manager

    def get_connections(self, forward: bool = True, repeat: bool = False) -> set[str]:
        if forward:
            if not repeat and self.value in self.connections:
                return self.connections - set(self.value)
            else:
                return self.connections
        else:
            if not repeat and self.value in self.backwards:
                return self.backwards - set(self.value)
            else:
                return self.backwards

    def apply_backwards(self):
        """Connect all the self.backwards for this nodes connected items"""
        for i in self.connections:
            self.manager[i].register_backwards(self.value)

    def register_backwards(self, item: str):
        if item not in self.backwards:
            # Track number of unique ways this tag could be seen
            self.max_permutations += len(self.connections)
            self.backwards.add(item)

    def unique_ways(self):
        return self.max_permutations

    def djikstras(self, heap: list[tuple[int, tuple[str,...]]], path: tuple[str, ...], offset: int = 0, backwards: bool= True):
        """Adds all connected nodes (forwards or backwards) to a heap"""

        item = self.backwards if backwards else self.connections
        for i in item:
            heapq.heappush(heap, (1+offset, path + (i, )))


class Word:
    def __init__(self, name:str, tag_list: list[str], manager: POS_manager):
        self.name = name
        self.tag_list = tag_list
        self.tag_probability: list[float] = [0 for _ in tag_list]
        self.locked = False
        self.seen_connections: dict[str,dict[tuple[str, str],float]] = {}
        self.manager = manager

    def update_tag(self, tag: str, before_after: tuple[str, str], probability: float):
        tag_dict = self.seen_connections[tag]
        
        if before_after in tag_dict:
            # Take the highest seen probability
            tag_dict[before_after] = max(probability, tag_dict[before_after])
        else:
            tag_dict[before_after] = probability
        # Set the probability of something being a certain pos via this formula
        # The sum of all probabilities that it was in a given context, divided by how many contexts exist
        self.tag_probability[self.tag_list.index(tag)] = sum(tag_dict.values()) / self.manager[tag].unique_ways()
        #TODO: Promote words to a LockedWord if applicable


    def query_type(self) -> str:
        return choices(self.tag_list, self.tag_probability)[0]

    def __bool__(self) -> bool:
        """If the word is locked or not"""
        return False

    def __str__(self):
        return self.name


class LockedWord(Word):
    def __init__(self, name: str, tag: str, manager):
        super().__init__(name, [], manager)
        self.tag = tag

    def query_type(self) -> str:
        return self.tag

    def __bool__(self) -> bool:
        return True