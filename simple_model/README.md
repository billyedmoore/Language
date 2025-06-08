# Simple NN

## Idea

+ Tokens represent characters as index's within a dictionary.
+ There is a special token for padding.
+ Predict an output token.

**Input**: list of *n* tokens.
**Output**: predicted next token.

## Input Data

For testing and development I am using `The Poetical Works of William Wordsworth — Volume 3 (of 8) by William Wordsworth`
from [Project GutenBerg](https://www.gutenberg.org/).

I will select random characters from the input text. I will then randomly choose a number of characters of context to give.
