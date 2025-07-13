import logging
import os
import json

LOGGER = logging.getLogger(__name__)



            


class POS:
    def __init__(self):
        self.pos_tags = {}
        self.group_tags = {}
        self.part_tags = {}
        self.known_tags = set()
        self.known = {}

        self.init_pos_tags()
    
    def init_pos_tags(self):

        dir = os.path.join(os.path.dirname(__file__), 'POS_tags')
        LOGGER.debug(f'Loading POS tags from {dir}')

        for file in os.listdir(dir):
            if file.endswith('.json'): # Grabbing only jsons
                with open(os.path.join(dir, file), 'r') as f:
                    data = json.load(f)
                    match file:
                        case f if f.startswith('POS_'):
                            self.pos_tags[data["name"]] = data["rules"]
                            LOGGER.debug(f'Loading POS tag {data["name"]}')
                        
                        case f if f.startswith('PART_'):
                            self.part_tags[data["name"]] = data
                            LOGGER.debug(f"Loaded PART {data['name']}")
                        
                        case f if f.startswith('GROUP_'):
                            self.group_tags[data["name"]] = data
                            LOGGER.debug(f"Loaded GROUP {data['name']} (containing {data["contains"]})")
                    

        if len(self.pos_tags) == 0:
            LOGGER.warning(f'No POS tags found in {dir}')
            
        LOGGER.info(f'Loaded {len(self.pos_tags)} POS tags, {len(self.group_tags)} group tags, and {len(self.part_tags)} part tags from {dir}')

    def parse_sentence(self, sentence: list[str]):
        if len(sentence) == 1 and sentence[0] not in self.known_tags:
            :
