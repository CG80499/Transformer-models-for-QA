from glob import glob
import random
from collections import Counter
import torch
import numpy, csv, gc
from tqdm import tqdm
import transformers
from datasets import load_dataset

QA_dataset = load_dataset("nq_open")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Running on: "+str(device).upper())

if torch.cuda.is_available():
    print("Clearing CUDA cache...")
    gc.collect()
    torch.cuda.empty_cache()

tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M", use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
vocab = tokenizer.get_vocab()
index_to_vocab = dict(zip(vocab.values(), vocab.keys()))

def process_answer(answer):
    if len(answer) == 1:
        return answer[0]
    else:
        return ", ".join(answer)

class question_dataset(torch.utils.data.Dataset):
    
    def __init__(self, questions, alpha=0.1):
        self.questions = questions
        self.embedding_size = 50257
        self.alpha = alpha
        self.maxlen = 36
        self.x = []


    def __len__(self): 
        return len(self.questions)
    
    def __getitem__(self, idx): 
        question = self.questions[idx]
        answer_str = question["answer"][0]
        input_sentence = tokenizer("<|endoftext|>"+question["question"]+"?"+answer_str, padding='max_length', truncation=True, return_tensors="pt", max_length=self.maxlen)
        tokens_in_question = len(tokenizer.tokenize(question["question"]+"?"))
        answer = tokenizer.tokenize(question["question"]+"?"+answer_str)
        output = torch.zeros((self.maxlen, self.embedding_size))
        words_in_output = 0
        for token in answer[:self.maxlen]:
            output[words_in_output, vocab[token]] = 1
            words_in_output += 1

        output = (1 - self.alpha)*output + self.alpha / self.embedding_size
        #words_in_output = min(10, words_in_output)
        mask = torch.zeros(self.maxlen)
        mask[tokens_in_question:words_in_output] = torch.ones(self.maxlen)[tokens_in_question:words_in_output]
        assert not sum(mask) == 0, answer_str
        return input_sentence["input_ids"], input_sentence["attention_mask"], output.type(torch.FloatTensor), mask.type(torch.FloatTensor)

QAdataset_train = list(filter(lambda x: len(x["answer"]) == 1 and len(x["answer"][0]) > 1, QA_dataset["train"]))
QAdataset_val = list(filter(lambda x: len(x["answer"]) == 1 and len(x["answer"][0]) > 1, QA_dataset["validation"]))
del QA_dataset

dataset = question_dataset(QAdataset_train)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

dataset_val = question_dataset(QAdataset_val) 
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=8, shuffle=True, num_workers=2)

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        print("Loading GPT-NEO model...")
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

model = Model().to(device)
#model = torch.load("GPT_NEO_QA_V1.pt")
print("Model loaded.")
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)      
loss_fn = torch.nn.KLDivLoss(reduction="none")

def get_loss(y_pred, y, mask):
    loss = loss_fn(y_pred, y) #input FIRST, output LAST
    loss = torch.sum(loss, 2)
    loss = (torch.sum(loss*mask, 1)/torch.sum(mask, 1)).mean()
    return loss

def get_accuracy(y_pred, y, mask):
    acc = (torch.sum(torch.argmax(y, dim=2)*mask==torch.argmax(y_pred, dim=2)*mask, 1)-(mask.shape[1] - torch.sum(mask, 1)))/torch.sum(mask, 1)
    return acc.mean().item()

train_size = len(dataset)

print("Data loaded and ready.")

epochs = 1000
best_accuracy = 0
min_loss = 2
max_batches = 20
print("Dataset size: "+str(train_size))

for n in range(epochs):
    losses = []
    accuracies = []
    
    for i, (input_ids, attention_mask, Y, mask) in tqdm(enumerate(dataloader), total=train_size//8+1, desc="Epoch "+str(n+1)):
        input_ids, attention_mask, Y, mask = input_ids.to(device), attention_mask.to(device), Y.to(device), mask.to(device)
        Y_pred = model.forward(input_ids.squeeze(dim=1), attention_mask.squeeze(dim=1))
        loss = get_loss(Y_pred, Y, mask)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        optimizer.zero_grad()
        losses.append(float(loss))
        accuracies.append(get_accuracy(Y_pred, Y, mask))
        if i%50 == 0:
          torch.cuda.empty_cache()
          gc.collect()
        #if i == 100:
        #    break
    
    
    # Val Metrics 
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
      gc.collect()
    val_losses = []
    val_accuracies = []
    model.eval()
    print("Validating...")
    for i, (input_ids, attention_mask, val_Y, mask) in tqdm(enumerate(dataloader_val)):
        input_ids, attention_mask, val_Y, mask = input_ids.to(device), attention_mask.to(device), val_Y.to(device), mask.to(device)
        Y_pred = model.predict(input_ids.squeeze(dim=1), attention_mask.squeeze(dim=1))
        val_loss = get_loss(Y_pred, val_Y, mask)
        val_losses.append(float(val_loss))
        val_accuracies.append(get_accuracy(Y_pred, val_Y, mask))
        if i%50 == 0:
          torch.cuda.empty_cache()
          gc.collect()

    model.train()
    val_loss = sum(val_losses)/len(val_losses)
    val_accuracy = sum(val_accuracies)/len(val_accuracies)
    avg_loss = sum(losses)/len(losses)
    accuracy = sum(accuracies)/len(accuracies)
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save(model, 'GPT_NEO_MAX_QA_V2.pt')
        print("Saved best val!")
    print("Epoch number: {} == Loss: {} == Val Loss: {}".format(n+1, avg_loss, val_loss))
    print("Epoch number: {} == Accuracy: {} == Val Accuracy: {} == Best Val Accuracy: {}".format(n+1, accuracy, val_accuracy, best_accuracy))