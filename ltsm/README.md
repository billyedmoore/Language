# LTSM (Long Short-term Memory)

LTSM is a kind of RNN with a more complex structure better suited to long-term memory.

## Idea

+ Tokens represent characters.
+ Predict an output token from a single input token.
+ Context is stored in a hidden layer, so prediction includes context from the previous tokens.

**Input**: one-hot encoded token and two context layers hidden layer and cell layer.
**Output**: predicted next tokens (probabilities) and an updated hidden and cell layer.

## Input Data

+ For each input we randomly select a start_index, we we select `n` characters from this index.
+ For `X` we have the tokens one-hot-encoded, for `Y` we have the next token for each input (not one-hot-encoded).

## The Model

+ LTSM with one layer and a single linear layer.
+ Hidden layer and cell layer are of (the same) arbitrary size.

