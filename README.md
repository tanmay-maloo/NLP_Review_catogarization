# NLP_Review_catogarization
Preprocessing of the text data and tring of different RNN and Birt models for better results on test sets.

## DIFFERENT algorithms were tried and accuracy on test set is compaired ##
**1.** `KNN` (k-nearest neighbour) treditional ML algorthm 

**2.** `RNN`-(Recurrent Neural Network) with 4 LSTM layers with each having 0.2 droput

**3.** `Bert` model with 12 layers fold developed by google (12-layer, 768-hidden, 12-heads, 110M parameters)

**4.** `Bert2` model with 24 layers fold developed by google (24-layer, 1024-hidden, 16-heads, 340M parameters)
checkout:`https://github.com/google-research/bert` for more

accuracy on `test.csv` were **0.38, 0.55, 0.61, 0.63** respectively
