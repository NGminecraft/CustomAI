from Semantics.vocabulary import Vocabulary
from Model.model import Transformer
from constants import START_TOKEN, END_TOKEN, SINGLE_WORD_LOSS_FN, EMBEDDING_SIZE, SINGLE_WORD_FACTOR, MAX_OUTPUT_WORDS
import torch
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
        from constants import DUPLICATE_WORD_EXPONENT
        
        word, _, tgt = self.vocabulary.get_embedding_from_output(input)
        loss = SINGLE_WORD_LOSS_FN(input, tgt.reshape(1, EMBEDDING_SIZE))
        loss *= SINGLE_WORD_FACTOR
        
        if word == self.last_word:
            loss *= self.duplicate_count
            self.duplicate_count *= DUPLICATE_WORD_EXPONENT

        loss.backward()

    def train_full_output(self, input, input_tensors):
        LOGGER.debug("Training full output")
        """BASIC STRUCTURE"""

        """CLEANING"""
        cleaned_sentence = []
        for i in input:
            if i != START_TOKEN and i != END_TOKEN:
                cleaned_sentence.append(i)

        if len(cleaned_sentence) == 0:
            LOGGER.error("The model outputed no real characters")
            return
        
        cleaned_sentence = " ".join(cleaned_sentence)

        LOGGER.debug(f"Output filtered to {cleaned_sentence}")

        """GRAMMAR"""
        from constants import GRAMMATICAL_LOSS

        corrected = self.language_checker.correct(cleaned_sentence) # Generate grammatically correct version
        LOGGER.debug(f'AI output corrected to "{corrected}"')

        # Update associations
        self.vocabulary.parse_sentence(corrected) # TODO: Give this parse a higher weight

        grammar_loss = GRAMMATICAL_LOSS(input_tensors, self.vocabulary.embed_sentence(corrected))
        
        # If the model output was really bad, raise loss by two
        LOGGER.debug(f"Grammatical loss was {grammar_loss}")

        from constants import GRAMMAR_WEIGHT
        total_loss = (
            grammar_loss * GRAMMAR_WEIGHT
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
