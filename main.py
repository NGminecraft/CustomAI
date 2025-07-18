import Model.AI as ai
from Semantics.POS.POS import interactive
import constants
import argparse
import logging

LOGGER = logging.getLogger()

parser = argparse.ArgumentParser()

parser.add_argument("-t", "--type", help= "What type to run", action="store", default="I")

parser_namespace = parser.parse_args()
run_type = parser_namespace.type[0].upper()

match run_type:
    case "I":
        LOGGER.info("Running the interactive test")
        ai.interact()
        
    case "P":
        LOGGER.info("Running the POS testing")
        interactive()

