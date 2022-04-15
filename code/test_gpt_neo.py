import transformers
import torch

tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M", use_fast=True)

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        gpt = transformers.AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
        gpt.train()
        self.gpt = gpt
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)
    
    def forward(self, input_ids, attention_mask):
        Y = self.gpt(input_ids= input_ids, attention_mask=attention_mask,return_dict=False)[0]
        Y = self.logsoftmax(Y)
        return Y
    
    def predict(self, input_ids, attention_mask):
        with torch.no_grad():
            Y = self.forward(input_ids, attention_mask)
        return Y

model = Model()
model = torch.load("GPT_NEO_MAX_QA_V2.5.pt", map_location=torch.device('cpu'))
"""
for child in model.gpt.children():
  for layer in child.modules():
    if isinstance(layer, torch.nn.Dropout):
        layer.p = 0.15
"""
#AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
model.eval()

tokenizer.pad_token = tokenizer.eos_token
vocab = tokenizer.get_vocab()
index_to_vocab = dict(zip(vocab.values(), vocab.keys()))
maxlen = 128
end_token_index = tokenizer.eos_token_id

interesting_questions = ["when was texas admitted to the union",
                        "when was texas admitted to the confederacy",
                        "which french king was killed in the french revolution",
                        "what is the tallest mountain in the world",
                        "when was the act of union between england and scotland signed",
                        "In mathematics, who invented calculus",
                        "In physics, who discovered special relativity",
                        'when did the coronation of queen elizabeth takes place',
                        "what is the largest country in the world",
                        "who was the last Tsar",
                        'which soviet leader gets his name from "man of steel"',
                        "what is the fastest land animal",
                        "what is longest river in the world",
]
answers = []
for q in interesting_questions:
    input_sentence = tokenizer('<|endoftext|>'+q+'?', padding='max_length', truncation=True, return_tensors="pt", max_length=maxlen)
    length = torch.sum(input_sentence["attention_mask"][0]).item()

    i = 0
    while not input_sentence["input_ids"][0][length-1] == end_token_index and i < 6:
        length = torch.sum(input_sentence["attention_mask"][0]).item()

        Y = model.predict(input_ids=input_sentence["input_ids"], attention_mask=input_sentence["attention_mask"])
        Y = torch.argmax(Y, dim=-1)[0]

        #print(input_sentence["input_ids"][0].shape)

        input_sentence["input_ids"][0][length] = Y[length-1]
        input_sentence["attention_mask"][0][length] = 1


        output = [index_to_vocab[i.item()] for i in input_sentence["input_ids"][0][0:length+1]]
        #print(" ".join(output))
        i += 1
        

    print(tokenizer.decode(input_sentence["input_ids"][0][0:length+1]))
    answers.append(tokenizer.decode(input_sentence["input_ids"][0][0:length+1]).split("?")[1])

print(answers)
