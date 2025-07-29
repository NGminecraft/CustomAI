import logging
import os
import json
from Semantics.POS.POS_graph import POS_manager, POSGraph

LOGGER = logging.getLogger(__name__)
JSON_FILE = "POS_tags\\pos_data.json"

def load_json():
    path = os.path.join(os.path.dirname(__file__), JSON_FILE)
    if not os.path.exists(path):
        LOGGER.critical(f"JSON file '{JSON_FILE}' not found in {path}")
        raise FileNotFoundError(f"JSON file '{JSON_FILE}' not found.")

    with open(path, 'r') as file:
        try:
            return json.load(file)
        except json.JSONDecodeError:
            LOGGER.critical(f"Error decoding JSON from file '{JSON_FILE}'")
            raise ValueError(f"Error decoding JSON from file '{JSON_FILE}'")


class POS:
    def __init__(self):
        self.json_data = load_json()
        self.POS = set()
        self.POS_manager = POS_manager()
        self.verify_json()

    def verify_json(self):
        from constants import START_TOKEN, END_TOKEN
        assert "tags" in self.json_data
        self.POS = set(self.json_data.get("tags"))
        discovered_tags = set()

        for tag, connections in self.json_data.items():
            match tag:
                case "tags":
                    continue
                case "required":
                    for i in connections:
                        self.POS_manager.add_required_feature(set(i))

                case a:
                    if a == "start":
                        a = START_TOKEN
                        discovered_tags.add("start")
                        self.POS.remove("start")
                        self.POS.add(START_TOKEN)
                    if "end" in connections:
                        connections[connections.index("end")] = END_TOKEN 
                    discovered_tags.add(a)
                    self.POS_manager.add_tag(a, connections)

        self.POS_manager.add_tag(END_TOKEN, [])

        for tag in self.POS - discovered_tags:
            LOGGER.warning(f"POS tag '{tag}' not found in JSON data. It will not be managed by POS_manager.")

        self.POS_manager.establish(END_TOKEN, START_TOKEN)
        self.POS_manager.match_sentence(["[<SOS>]", "Hello", "!", "[<EOS>]"])
    
    def match_sentence(self, sentence):
        return self.POS_manager.match_sentence(sentence)

def interactive():
    from constants import SENTENCE_SPLITTER, START_TOKEN, END_TOKEN
    pos = POS()

    while True:
        a = input("Enter a sentence (or exit): ")
        if a == "exit":
            break
        
        sentence = START_TOKEN+" "+a+" "+END_TOKEN
        sentence = sentence.upper()

        split_sentence = SENTENCE_SPLITTER(sentence)
        result = pos.match_sentence(split_sentence)

        if result is None:
            break

        matches, prob = result
        matches = [str(i) for i in matches]
        prob = [str(i) for i in prob]
        padded_matches, padded_prob, padded_result = [], [], []
        for i, v in enumerate(matches):
            section_length = max(len(v), len(prob[i]), len(result)) + 2
            before = " " * ((section_length - len(v)) // 2)
            after = " " * ((section_length - len(v)) - len(before))
            padded_matches.append(before + v + after)
            
            before = " " * ((section_length - len(prob[i])) // 2)
            after = " " * ((section_length - len(prob[i])) - len(before))
            padded_prob.append(before + prob[i] + after)

            before = " " * ((section_length - len(split_sentence[i])) // 2)
            after = " " * ((section_length - len(split_sentence[i])) - len(before))
            padded_result.append(before + split_sentence[i] + after)

        from os import get_terminal_size
        distance_traversed = 0
        match_count = 0
        max_distance = get_terminal_size().columns
        print()

        def flush_prob(prob: tuple[list[str], ...], count: int):
            for i in prob:
                for _ in range(count):
                    print(i.pop(0), end=" ")
                print()


        while len(padded_prob) > 0:
            if len(padded_result) == 0:
                print()
                flush_prob((padded_matches, padded_prob), match_count)
                print()
                break

            first_match = padded_result.pop(0)
            if max_distance > len(first_match) + 1 + distance_traversed:
                print(first_match, end=" ")
                match_count += 1
            else:
                print()
                flush_prob((padded_matches, padded_prob), match_count)
                print("\n" * 2)
                match_count = 1
                distance_traversed = 0
                print(first_match, end=" ")
            
            distance_traversed += len(first_match) + 1
