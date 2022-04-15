# Training Data
I trained both models on Google's EfficientQA dataset(\https://huggingface.co/datasets/nq_open). However, I removed questions with multiple answers. These questions account for less than 10% of the dataset and significantly increase training time while reducing model performance.

# Training 
I used label smoothing with an alpha value of 0.1 for both models. I used the pointwise KL-divergence loss function to measure the difference between the actual and predicted distributions. I used AdamW with a learning rate of 1e-5 and 2e-5 for BERT and GPT respectively and a batch size of 8. 

# Bert 66M
I choose to use the distil BERT variant. I used masking to compute the loss/accuracy only on the answer tokens. Due to BERT not a being generative model, I found issues relating to broken English or the model repeating tokens in the questions were common. To mitigate this I removed all duplicate tokens and all tokens in the questions from the answer.

# GPT Neo 125M
Similarly, to BERT reducing the max length to 35 tokens means that training is fairly quick. The main limitations are the amount of GPU RAM you have. I tried 3 different loss computation strategies. 
1. Compute loss/accuracy on all question/answer tokens.
2. Compute loss/accuracy on just the answer tokens with the end token.
3. Compute loss/accuracy on just the answer tokens without the end token.

The nice thing about strategy 1 is that the model learns to predict the questions before you have asked them. But only ~10% of tokens are answer tokens, this resulted in plausible-sounding questions being generated but with completely incorrect answers. In strategy 2, 30-50% of all tokens are the end token giving poor model performance. I found strategy 3 while limiting the answer lenght to 6 tokens to be most effective as the model is strongly penalised for wrong answers. I think a version of strategy 1 could be best if you weight the value of question and answer tokens such that the loss is 50% question tokens and 50% answer tokens.

# Results
Here the Results from a set of test questions I divised. For reference, I included GPT 3 Ada which has 2.7B parameters(without fine-tuning or prompting).


