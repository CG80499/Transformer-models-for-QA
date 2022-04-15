import torch
import transformers
from datasets import load_dataset

QA_dataset = load_dataset("nq_open")

QAdataset_train = list(filter(lambda x: len(x["answer"]) == 1, QA_dataset["train"]))
QAdataset_val = list(filter(lambda x: len(x["answer"]) == 1, QA_dataset["validation"]))
del QA_dataset

tokenizer = transformers.AutoTokenizer.from_pretrained("distilbert-base-uncased")
vocab = tokenizer.get_vocab()
index_to_vocab = dict(zip(vocab.values(), vocab.keys()))

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        bert = transformers.AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")
        #bert.eval()
        self.bert = bert
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)
    
    def forward(self, input_ids, attention_mask):
        Y = self.bert(input_ids= input_ids, attention_mask=attention_mask,return_dict=False)[0]
        Y = self.logsoftmax(Y)
        return Y
    
    def predict(self, input_ids, attention_mask):
        with torch.no_grad():
            Y = self.forward(input_ids, attention_mask)
        return Y

model = Model()
for param in model.bert.parameters():
    param.requires_grad = False
for param in model.parameters():
    param.requires_grad = False
model = torch.load("BERT_QA_V2.pt", map_location=torch.device('cpu'))
model.eval()

def process(q):
    return q[0].upper()+q[1:]+"?"

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
    question_index = 13
    QAdataset_val[question_index]["question"] = q
    print("Question: ", QAdataset_val[question_index]["question"])
    #print("Answer: ", QAdataset_val[question_index]["answer"][0])
    #print("Input sentence: ", QAdataset_val[question_index]["question"]+"? [SEP]"+" [MASK]"*10)
    input_sentence = tokenizer(QAdataset_val[question_index]["question"]+"? [SEP]"+" [MASK]"*10, padding='max_length', truncation=True, return_tensors="pt")
    input_tokens = tokenizer.tokenize(QAdataset_val[question_index]["question"]+"? [SEP]"+" [MASK]"*10, padding='max_length', truncation=True, return_tensors="pt")
    output = model.predict(input_sentence["input_ids"], input_sentence["attention_mask"])

    Y = torch.argmax(output[0], dim=-1)
    l = len(QAdataset_val[question_index]["question"]+"? [SEP]")
    output = [index_to_vocab[i.item()] for i in Y][0:l+10]
    print(" ".join(output))
    final_output = []
    for i, e in enumerate(output):
        if e not in final_output and e not in input_tokens and e not in [")", "(", "[SEP]", ".", "!", "?", ","]:
            final_output.append(e)
    final_output = " ".join(final_output)
    final_output = final_output.replace(" ##", "").replace("##", "")
    #final_output = final_output[0].upper()+final_output[1:]+"."
    print(final_output)
    answers.append(final_output)
print(answers)


