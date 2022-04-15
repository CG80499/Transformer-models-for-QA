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

Question: When was texas admitted to the union?
Bert (66M): december march 1845 (mostly correct)
GPT-Neo (125M): 1898-1901. (false)
GPT-3 Ada (2.7B): texas was admitted to the union on january 6, 1907. (false)

Question: When was texas admitted to the confederacy?
Bert (66M): december 1861 (mostly correct)
GPT-Neo (125M): 1821-1822, (false)
GPT-3 Ada (2.7B): texas was admitted to the confederacy on march 14, 1856.(false)

Question: Which french king was killed in the french revolution?
Bert (66M): louis xvi august (mostly correct)
GPT-Neo (125M): Louis XVI of France, Prince (correct)
GPT-3 Ada (2.7B): The French king, France, was assassinated in the French revolution.(false)

Question: What is the tallest mountain in the world?
Bert (66M): 2 everest mount (mostly correct)
GPT-Neo (125M): Mount Everest in Nepal, Everest (correct)
GPT-3 Ada (2.7B): Mount Everest, the worlds tallest mountain, is 5,853 feet (1,869 meters).(correct)

Question: When was the act of union between england and scotland signed?
Bert (66M): 1707 january (mostly correct)
GPT-Neo (125M): 1707-1708. (mostly correct)
GPT-3 Ada (2.7B): The act of union between England and Scotland was on October 14, 1492.(false)

Question: In mathematics, who invented calculus?
Bert (66M): of he introduced aristotle (false)
GPT-Neo (125M): John Maynard Keynes, Jr (false)
GPT-3 Ada (2.7B): The study of calculus is known as mathematics.(false)

Question: In physics, who discovered special relativity?
Bert (66M): he johann hans wilhelm einstein (somewhat correct)
GPT-Neo (125M): John Wheeler's theory of general (false)
GPT-3 Ada (2.7B): Sir Arthur Conan Doyle discovered special relativity in Doyles life time, in 1892 when he published a paper in the Journal of the Royal Society 
of London about the laws of motion and energy in a three-dimensional space.(false)

Question: When did the coronation of queen elizabeth takes place?
Bert (66M): 1 6 day june 25 1952 ii (false)
GPT-Neo (125M): 1928-1929 and (false)
GPT-3 Ada (2.7B): The coronation of queen elizabeth took place on April 14, 1499.(false)

Question: What is the largest country in the world?
Bert (66M): russia (correct)
GPT-Neo (125M): Russia and China, with the (mostly correct)
GPT-3 Ada (2.7B): There is no definitive answer to this question as there are a variety of opinions as to the largest country in the world. Some people believe that the largest country is China, while others believe that the largest country is the United States. Ultimately, the largest country in the world will be determined by the number of population points that are reached.(false)

Question: Who was the last Tsar?
Bert (66M): nicholas ivan ii mikhail 1917 (vaguely correct)
GPT-Neo (125M): Chris Rockwell Jr. and (false)
GPT-3 Ada (2.7B): The last Tsar was Nicholas I of Russia.(mostly correct)

Question: Which soviet leader gets his name from "man of steel"?
Bert (66M): the got stalin joseph mikhail lenin iron(vaguely correct)
GPT-Neo (125M): Karl Marx, Jr. of (false)
GPT-3 Ada (2.7B): The soviet leader's name is  Fidel Castro .(false)

Question: What is the fastest land animal??
Bert (66M): or indian eagle (false)
GPT-Neo (125M): The cheetah dolphin (mostly correct)
GPT-3 Ada (2.7B): The fastest land animal is a land animal that can travel at a fast pace. This is a land animal that can travel at a fast pace.(false)

Stats:
Bert (66M): 75.0%
GPT-Neo (125M): 41.67%
GPT3 (125M): 16.67%
