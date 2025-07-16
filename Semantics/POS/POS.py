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
        self.POS.remove("start")
        self.POS.add(START_TOKEN)
        discovered_tags = set()

        for tag, connections in self.json_data.items():
            match tag:
                case "tags":
                    continue
                case additional if tag not in self.POS:
                    pass

                case a:
                    if a == "start":
                        a = START_TOKEN
                    if "end" in connections:
                        connections[connections.index("end")] = END_TOKEN 
                    discovered_tags.add(a)
                    self.POS_manager.add_tag(a, connections)

        self.POS_manager.add_tag(END_TOKEN, [])

        for tag in self.POS - discovered_tags:
            LOGGER.warning(f"POS tag '{tag}' not found in JSON data. It will not be managed by POS_manager.")

        self.POS_manager.establish(END_TOKEN, START_TOKEN)
