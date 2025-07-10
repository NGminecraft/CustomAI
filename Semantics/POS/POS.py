import logging

LOGGER = logging.getLogger(__name__)


def init_pos_tags():
    import os
    import json
    pos_tags = {}
    dir = os.path.join(os.path.dirname(__file__), 'POS_tags')
    LOGGER.debug(f'Loading POS tags from {dir}')
    for file in os.listdir(dir):
        if file.endswith('.json') and file.startswith('POS_'):
            with open(os.path.join(dir, file), 'r') as f:
                LOGGER.debug(f'Loading POS tag {file[4:-3]}')
                pos_tags[file[4:-3]] = json.load(f)
    if len(pos_tags) == 0:
        LOGGER.warning(f'No POS tags found in {dir}')
        return {}
    LOGGER.info(f'Loaded {len(pos_tags)} POS tags from {dir}')
    return pos_tags
            


class POS:
    def __init__(self):
        self.POS_TAGS = init_pos_tags()