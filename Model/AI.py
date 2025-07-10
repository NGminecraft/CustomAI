from Semantics.vocabulary import Vocabulary
from Model.model import Transformer
from constants import START_TOKEN, END_TOKEN, SINGLE_WORD_LOSS_FN, EMBEDDING_SIZE, SINGLE_WORD_FACTOR, MAX_OUTPUT_WORDS
import torch
import torch.nn as nn
import logging
import language_tool_python
from language_tool_python.utils import classify_matches, TextStatus

LOGGER = logging.getLogger(__name__)

class AI:
    def __init__(self):
        self.vocabulary = Vocabulary()
        self.vocabulary.parse_sentence(START_TOKEN)
        self.vocabulary.parse_sentence(END_TOKEN)
        self.model = Transformer().cuda()
        self.language_checker = language_tool_python.LanguageTool('en-US')
        self.last_word = None
        self.duplicate_count = 1
        self.started = False

    def __call__(self, input_sentence, full=True, train=True) -> str:
        LOGGER.info(f"Recieved input {input_sentence}")
        # Give the start and end signal tokens
        completed = False
        full_word_output = []
        full_tensor_output = None
        input_sentence = START_TOKEN +" "+ input_sentence +" "+ END_TOKEN

        # Update vocab
        self.vocabulary.parse_sentence(input_sentence)

        # Generates the key that defines the embedding
        model_input = self.vocabulary.embed_sentence(input_sentence)

        # Switching to a tensor object
        model_input.requires_grad = True
        model_input.unsqueeze(0)
        
        # Creates the target tensor with the start token
        tgt = self.vocabulary.get_embedding(START_TOKEN).unsqueeze(0).unsqueeze(0)

        for iterations in range(MAX_OUTPUT_WORDS):
            if completed:
                LOGGER.debug(f"Model finished in {iterations + 1} iterations")
                break
            
            # Forward pass through the model
            output = self.model.forward(model_input, tgt)

            next_token = output[:, -1, :]

            word, _, word_tensor = self.vocabulary.get_embedding_from_output(next_token)

            if train:
                self.train_word(next_token)
            
            if not full:
                return word
            
            full_word_output.append(word)

            if full_tensor_output is None:
                full_tensor_output = word_tensor
                full_tensor_output.requires_grad = True
            else:
                full_tensor_output = torch.vstack((full_tensor_output, word_tensor))
            
            if word == END_TOKEN:
                completed = True

            tgt = torch.cat((tgt, word_tensor.reshape(1, 1, EMBEDDING_SIZE)), dim=1)
        else:
            LOGGER.error("Model hit the word count")

        if train:
            self.train_full_output(full_word_output, full_tensor_output)

        LOGGER.debug(f"Model evaluated to {full_word_output}")
        
        return " ".join(full_word_output)
            

    def train_word(self, input):
        from constants import DUPLICATE_WORD_EXPONENT, DUPLICATE_START
        
        word, _, tgt = self.vocabulary.get_embedding_from_output(input)
        loss = SINGLE_WORD_LOSS_FN(input, tgt.reshape(1, EMBEDDING_SIZE))
        loss *= SINGLE_WORD_FACTOR
        
        if word == self.last_word:
            loss *= self.duplicate_count
            self.duplicate_count *= DUPLICATE_WORD_EXPONENT
        if word == START_TOKEN:
            if not self.started:
                self.started = True
            else:
                loss += DUPLICATE_START

        loss.backward()

    def train_full_output(self, input_words, input_tensors):
        from constants import GRAMMATICAL_LOSS, NO_OUT_WEIGHT

        LOGGER.debug("Training full output")
        """BASIC STRUCTURE"""

        """CLEANING"""
        cleaned_sentence = []
        for i in input_words:
            if i != START_TOKEN and i != END_TOKEN:
                cleaned_sentence.append(i)

        if len(cleaned_sentence) == 0:
            LOGGER.error("The model outputed no real characters")
            loss = GRAMMATICAL_LOSS(input_tensors, self.vocabulary.embed_sentence(START_TOKEN + END_TOKEN)) * NO_OUT_WEIGHT
            loss.backward()
            return
        
        cleaned_sentence = " ".join(cleaned_sentence)

        LOGGER.debug(f"Output filtered to {cleaned_sentence}")

        """GRAMMAR"""

        corrected = self.language_checker.correct(cleaned_sentence) # Generate grammatically correct version
        LOGGER.debug(f'AI output corrected to "{corrected}"')

        # Update associations
        self.vocabulary.parse_sentence(corrected) # TODO: Give this parse a higher weight

        corrected = START_TOKEN + " " + corrected + " " + END_TOKEN # Add start and end tokens

        output_loss = GRAMMATICAL_LOSS(input_tensors, self.vocabulary.embed_sentence(corrected))
        
        # If the model output was really bad, raise loss by two
        LOGGER.debug(f"Grammatical loss was {output_loss.item()}")

        try:
            user_loss = int(input(f"AI_OUTPUT: {cleaned_sentence}\nScale from  to 10? ")) - 10 * -1
        except ValueError:
            LOGGER.warning("Invalid input for user loss, defaulting to 0")
            user_loss = 0

        from constants import OUTPUT_WEIGHT, USER_LOSS
        total_loss = (
            output_loss * OUTPUT_WEIGHT + 
            user_loss * USER_LOSS
        )

        LOGGER.debug(f"Total loss was {total_loss.item()}")
        total_loss.backward()

    def __del__(self):
        self.language_checker.close()


def main():
    a = AI()
    val = a("Hello World")

    print(val)

def interact():
    a = AI()
    while True:
        input_sentence = input("Enter a sentence (or 'exit' to quit): ")
        if input_sentence.lower() == 'exit':
            break
        output = a(input_sentence)
        print(f"AI Response: {output}")
