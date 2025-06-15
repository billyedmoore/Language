# Simple NN

## Idea

+ Tokens represent characters.
+ There is a special token for padding.
+ Predict an output token from a series of input tokens.

**Input**: list of *n* one-hot encoded tokens.
**Output**: one-hot encoded predicted next token (probablities).

## Input Data

+ Input is a concatonated series of tokens of `n` length.
+ For each input we randomly sample the target character from the input text.
+ We randomly select a length of previous context to give `> 1` and `< n`. 
+ The previous context is padded out to `n` length 
+ Tokens are one-hot-encoded and concatonated into a 1d vector of `n*c` length where
  `c` is the number of unique chars in the text.

## The Model

+ Simple NN with 2 linear layers.
+ `n*c -> [linear_1] -> (n*c)//2 -> [linear_2] -> c`
+ We use weighting to reduce the impact of more frequent characters (to prevent a case
where the model only returned " ".

## Conclusion

The model struggles.
