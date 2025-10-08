import numpy as np

# new tokens for masking, deck size is 70 
MASK_TOKEN_ID = 70

def adapt_batch_for_encoder(batch):
    # TODO: take a batch, combine context and question
    # replace target token with MASK_TOKEN_ID
    # return model_input and the target_label
    pass