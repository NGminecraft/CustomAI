import logging
from random import choices, sample
import heapq

LOGGER = logging.getLogger(__name__)

class POS_manager:
    def __init__(self):
        self.tags: list[str] = []
        self._pos_tags: dict[str, POSGraph] = {}
        self._words: dict[str, Word|LockedWord] = {}
        self.initialized: bool = False
        self.distance_from_end = {}
        self.distances_to_path = {}
        self.end_token: str = ""
        self.start_token: str = ""
        self.required_features: list[set[str]] = []
        self.required_set: set[str] = set()
    
    def add_required_feature(self, feature: set[str]):
        self.required_features.append(feature)
        self.required_set.union(feature)

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

        self.end_token = end_token
        self.start_token = start_token
        self.initialized = True
    
    def match_sentence(self, sentence: list[str]) -> tuple[list[str], list[float]] | None:
        from itertools import chain

        def calculate_path_between(current_tag: str, cur_distance: int, target_tag: str, tgt_distance: int, all_of_these: set[str] = set(), one_of_these: set[str] | None = None) -> tuple[tuple[str, ...], tuple[float|int, ...]]:
            from random import random
            # This calculates some path between the start and end tags, prioritizing probable paths
            current_node = self._pos_tags[current_tag]
            # This heap stores the probability, the distance, some random value (adds random element to break ties, and stops the heap from comparing the graph objects)
            # As well as the path up to that object, and the chance of each item to get there
            heap: list[tuple[float, int, float, list[POSGraph], tuple[float|int]]] = [(-1, cur_distance - 1, 0, [current_node], (1, ))]

            while len(heap) > 0:
                # Grab the most probable (or shortest distance for ties), and the word its connected to
                probability, cur_distance, _, path_list, path_chance = heapq.heappop(heap)
                word_class = word_class_sentence[cur_distance+1]

                match cur_distance:
                    # We aren't there yet, so update the heap
                    case a if a < tgt_distance:
                        # Add the connections to the heap, prioritizing priority
                        for i in path_list[-1].connections:
                            if word_class.all_zero():
                                # If the word only has zero probabilities, then define probability by how many other tags could appear
                                prob = 1 / len(path_list[-1].connections)
                            else:
                                prob = word_class.get_probability(i)
                            heapq.heappush(heap, (prob * probability, cur_distance + 1, random(), path_list + [self._pos_tags[i]], path_chance + (prob, )))

                    case tgt_distance:
                        path_set = set(path_list)
                        # If we find one, if we don't then just leave it alone, any further additions to the heap will be to far
                        if path_list[-1].value == target_tag and all_of_these.issubset(path_set) and (one_of_these is not None and one_of_these.intersection(path_set)):
                            return tuple(str(i) for i in path_list[1:-1]), path_chance
                    
                
            LOGGER.error(f"The sentence from index {cur_distance} to {tgt_distance} failed to parse")
            return ("ERR", ) * (tgt_distance - cur_distance), (-1, ) * (tgt_distance - cur_distance)

        sentence_length = len(sentence)
        
        if sentence_length < 4: # Start token, interjection, punctuation, end token. Smallest sentence possible
            LOGGER.error(f"Sentence '{sentence}' has to few tokens!")
            return
        
        word_class_sentence: list[Word|LockedWord] = []
        locked_words: list[tuple[int, str]] = []
        # Get word classes for everything
        for i, v in enumerate(sentence):
            new_class = self.get_word_class(v)
            word_class_sentence.append(new_class)
    
            # Extract the locked words
            if new_class:
                locked_words.append((i, new_class.tag))

        current_distance = 0
        current_tag = sentence[0]
        # This just combines all the variables we track into one.
        # The tokens we guess, the probabilities of each token being that one, and which of the required words this fulfills
        result_sections: list[tuple[list[str], list[float], set[str]]] = []

        for path_distance, locked in locked_words:
            # Unpacking all our values
            guessed_tokens: list[str] = []
            probabilities: list[float] = []
            discovered_required: set[str] = set()

            # Set up the empty lists where data will be placed
            if current_distance != path_distance:
                # This part needs to calculate a path from the current node to the next locked ones (non-inclusive)
                final_match, probability = calculate_path_between(current_tag, current_distance, locked, path_distance)


                # add the tokens we're guessing to the final match, and update the current distance we've gone
                guessed_tokens.extend(final_match)
                probabilities.extend(probability)
                current_distance += len(final_match)
            
            # Include the locked word we are on
            probabilities.append(1.0)
            guessed_tokens.append(str(locked))
            current_distance += 1
            
            # Check to see if we hit any of the required features
            guessed_set = set(guessed_tokens)
            for i, v in enumerate(self.required_features):
                to_check = v.intersection(guessed_set)
                if to_check:
                    discovered_required.union(to_check)

            current_tag = guessed_tokens[-1]
            result_sections.append((guessed_tokens, probabilities, discovered_required))

        # Unpack all the required_features that we have found and place them into a set
        required_features_found = set().union(*(i for _, _, i in result_sections))

        # Did we get all the required features? (If the set of all found required features, and all required features is different)
        # Excludes sentences that can't fit all required tokens.
        # This will repeat until we do
        while len(required_features_found) != len(self.required_features) and sentence_length > 2 + len(self.required_features):
            LOGGER.debug("First attempt at POS identifying is missing features")
            min_heap = []
            idx = 0 
            # First we add all the tags to the heap. If we encounter any that don't fulfil any requiremtents, we try to get it too. 
            for tags, probs, required in result_sections:
                this_probability = sum(probs)/len(probs)
                
                # If there are no required items, attempt to get some
                if not required:
                    highest_probabilities = []
                    for i in self.required_features:
                        # Make sure we aren't trying to get a required tag we already have
                        if i.intersection(required_features_found):
                            continue
                        # The first part either uses the start_token if idx is zero, or takes the last tag in the previous result section 
                        match, prob = calculate_path_between(self.start_token if idx == 0 else result_sections[idx-1][0][-1], 1, tags[-1], 1 + len(tags), one_of_these=i)
                        
                        if match[0] == "ERR":
                            continue

                        heapq.heappush(highest_probabilities, (sum(prob)/len(prob) * -1, list(prob), set(match).intersection(self.required_features), list(match)))
                    
                    # Get the most likely alternative that does have one of the correct ones
                    _,  prob, discovered_required, match = heapq.heappop(highest_probabilities)
                    result_sections[idx] = (match, prob, discovered_required)
                    required_features_found.add(match)
                    continue
                else:
                    # Otherwise we just add it to a heap
                    heapq.heappush(min_heap, (this_probability * -1, idx))
                
            while len(min_heap) > 0:
                match, prob = calculate_path_between()

                idx += 1

        # Flatten the guessed tokens
        guessed_tokens = []
        probabilities = []
        for tokens, probs, _ in result_sections:
            guessed_tokens.extend(tokens)
            probabilities.extend(probs)

        chance_that_token: list[float|int] = [1]

        # Update the probability of all the words
        for i, v in enumerate(word_class_sentence[1:-1]):
            v.update_tag(guessed_tokens[i+1], (guessed_tokens[i], guessed_tokens[i+2]), probabilities[i+1])
            chance_that_token.append(v.get_probability(guessed_tokens[i+1]))
        
        chance_that_token.append(1)

        return guessed_tokens, chance_that_token
            

        
    def get_word_class(self, word):
        """Gets the word class for a word, or creates it if needed"""
        if word in self._words:
            return self._words[word]
        else:
            # Default non alpha-numeric words to punctuation
            if word == self.start_token or word == self.end_token:
                self._words[word] = LockedWord(word, word, self)
            elif not word.isalnum() and len(word) == 1:
                self._words[word] = LockedWord(word, "[<PUNCT>]", self)
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
    
    def __str__(self):
        return self.value

class Word:
    def __init__(self, name:str, tag_list: list[str], manager: POS_manager):
        self.name = name
        self.tag_list = tag_list
        self.tag_probability: list[float] = [0 for _ in tag_list]
        self.locked = False
        self.seen_connections: dict[str,dict[tuple[str, str],float]] = {i: {} for i in tag_list}
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

    def all_zero(self) -> bool:
        prob_set = set(self.tag_probability)
        return len(prob_set) == 1 and 0 in prob_set

    def get_probability(self, tag: str):
        return self.tag_probability[self.tag_list.index(tag)]

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
    
    def get_probability(self, tag: str):
        return 1
    
    def update_tag(self, tag: str, before_after: tuple[str, str], probability: float):
        return None
    
    def __bool__(self) -> bool:
        return True