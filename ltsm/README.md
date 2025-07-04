# LTSM (Long Short-term Memory)

LTSM is a kind of RNN with a more complex structure.

## Idea

+ Tokens represent characters.
+ Predict an output token from a single input token.
+ Context is stored in a hidden layer, so prediction includes context from the previous tokens.

**Input**: one-hot encoded token and a hidden layer.
**Output**: predicted next tokens (probabilities) and an updated hidden layer.

## Input Data

+ For each input we randomly select a start_index, we we select `n` characters from this index.
+ For `X` we have the tokens one-hot-encoded, for `Y` we have the next token for each input (not one-hot-encoded).

## The Model

+ Simple RNN with a single linear layer.
+ Hidden layer is of an arbitrary size.

## Conclusion

Performs much better than the naive linear approach, even producing something that kind of looks like words. 
If you ignore the full-stops that is.
