from math import e
import torch.nn as nn
import torch
from logging import config as logging_config
from Utilities.logger_colors import ColorFormatter


torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

"""LOGGING CONSTANTS"""
LOGGING_CONFIG = {
    'version': 1,
    'formatters': {
        'colored': {
            '()': ColorFormatter,
            'format': '%(levelname)s: %(message)s'
        },
        'standard': {
            'format': '[%(asctime)s] %(name)s.%(levelname)s: %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'colored',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.FileHandler',
            'level': 'DEBUG',
            'formatter': 'standard',
            'filename': 'app.log',
            'mode': 'w'
        }
    },

    'root': {
        'level': 'DEBUG',
        'handlers': ['console', 'file'],
    }
}
logging_config.dictConfig(LOGGING_CONFIG)


"""VOCABULARY CONSTANTS"""
START_TOKEN = "[<SOS>]"
END_TOKEN = "[<EOS>]"

WORD_RELATIONSHIP_WEIGHT = 0.1
SAME_WORD_VALUE = 1

LEARNING_FACTOR = 0.5
DECAY_FACTOR = 10
def DECAY_FN(x):
    return x - x * e ** (-DECAY_FACTOR * x)

DISTANCE_FACTOR = 1
def POSITION_AWARE_WEIGHT_FN(x):
    return 1 / (DISTANCE_FACTOR ** x)

WORD_SCALING_FACTOR = 0.93

def RELATIONSHIP_SCALING_FN(x):
    return x

SECONDHAND_RELATIONSHIP_WEIGHT = 0.3

STRING_SPLIT_REGEX = fr"\{START_TOKEN}|\{END_TOKEN}|\b[^\s\w]+|\b[^\s]+\b"
def SENTENCE_SPLITTER(sentence):
    import re
    return re.findall(STRING_SPLIT_REGEX, sentence, flags=re.IGNORECASE)

"""EMBEDDING CONSTANTS"""
EMBEDDING_SIZE = 100
EMBEDDING_NORMALIZATION = nn.Tanh()
PAD_IDX = 0

"""MODEL CONSTANTS"""
MAX_INPUT = 128
HEADS = 10
ENCODER_LAYERS = 6
DECODER_LAYERS = 6
FEEDFORWARD_SIZE = 2048

MAX_OUTPUT_WORDS = 1000

"""TRAINING CONSTANTS"""
DUPLICATE_WORD_EXPONENT = 3

_SINGLE_WORD_LOSS_FN = nn.CosineEmbeddingLoss(margin=0.3)
def SINGLE_WORD_LOSS_FN(output_embed, target_embed):
    return _SINGLE_WORD_LOSS_FN(output_embed, target_embed, torch.ones(EMBEDDING_SIZE))

SINGLE_WORD_FACTOR = 0.5 # Lower numbers = less drastic changes

_GRAMMAR_LOSS = nn.CosineEmbeddingLoss()
def GRAMMATICAL_LOSS(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    predicted_mean = predicted.mean(dim=0)
    target_mean = predicted.mean(dim=0)
    return _GRAMMAR_LOSS(predicted_mean.unsqueeze(0), target_mean.unsqueeze(0), torch.tensor([1]))



USER_LOSS = 5

NO_OUT_WEIGHT = 0.3
OUTPUT_WEIGHT = 3
