import pandas as pd
import numpy as np
from collections import defaultdict
import re
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification

import tensorflow as tf

log_dict = defaultdict(str)
log_list = []
block_list = []
with open("data/truncated_logs.txt") as f:
    for line in f:
        log_str = line.split()
        log_list.append(log_str)
        r = re.compile("blk_")
        block_id = list(filter(r.match, log_str))[0]

        if block_id in log_dict:
            log_dict[block_id] += log_str
        else:
            log_dict[block_id] = log_str

log_df = pd.DataFrame({"block_id": log_dict.keys(),"log": pd.arrays.SparseArray(log_dict.values())})
anomaly_df = pd.read_csv("data/HDFS_1/anomaly_label.csv")
merged_df = log_df.set_index('block_id').join(anomaly_df.set_index('BlockId'))
merged_df.dropna(inplace=True)
merged_df.replace({"Normal": 0, "Anomaly": 1}, inplace=True)
train_df = merged_df[:500]
test_df = merged_df.tail(500)
labels = train_df["Label"].to_numpy()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
logs = ["[CLS] " + " ".join(log)[-512:] + " [SEP]" for log in train_df["log"]]
tokenized_texts = [tokenizer.tokenize(log) for log in logs]
MAX_LEN = 128
input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts], maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
attention_masks = []
for seq in input_ids:
  seq_mask = [float(i>0) for i in seq]
  attention_masks.append(seq_mask)

train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, random_state=0, test_size=0.1)
train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=0, test_size=0.1)

# Convert all of our data into torch tensors, the required datatype for our model
train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)
train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

# Select a batch size for training. 
batch_size = 32

# Create an iterator of our data with torch DataLoader 
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

optimizer = BertAdam(optimizer_grouped_parameters, lr=2e-5, warmup=.1)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
  
training_loss_set = []
epochs = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for _ in range(epochs):  
    model.train()  
    training_loss = 0
    train_examples_count, train_steps_count = 0, 0
    for step, batch in enumerate(train_dataloader):
        print("step:", step)
        batch = tuple(t.to(device) for t in batch)
        batch_input_ids, batch_input_mask, batch_labels = batch
        optimizer.zero_grad()
        loss = model(
            batch_input_ids,
            token_type_ids=None,
            attention_mask=batch_input_mask,
            labels=batch_labels
        )
        training_loss_set.append(loss.item())
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
        train_examples_count += batch_input_ids.size(0)
        train_steps_count += 1

    print("Train loss: {}".format(training_loss/train_steps_count))
    
    model.eval()
    evaluation_loss, evaluation_accuracy = 0,0
    evaluation_steps_count, evaluation_examples_count = 0,0
    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        batch_input_ids, batch_input_mask, batch_labels = batch
        with torch.no_grad():
            logits = model(batch_input_ids, token_type_ids=None, attention_mask=batch_input_mask)    
        logits = logits.detach().cpu().numpy()
        label_ids = batch_labels.to('cpu').numpy()
        temp_eval_accuracy = flat_accuracy(logits, label_ids)    
        evaluation_accuracy += temp_eval_accuracy
        evaluation_examples_count += batch_input_ids.size(0)
        evaluation_steps_count += 1
    print("Validation Accuracy: {}".format(evaluation_accuracy/evaluation_steps_count))






# load test data
logs = ["[CLS] " + " ".join(log)[-512:] + " [SEP]" for log in test_df["log"]]
labels = test_df["Label"].to_numpy()

# tokenize test data
tokenized_texts = [tokenizer.tokenize(sent) for sent in logs]
# Pad our input tokens
input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
# Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
# Create attention masks
attention_masks = []
# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids:
  seq_mask = [float(i>0) for i in seq]
  attention_masks.append(seq_mask) 

# create test tensors
prediction_inputs = torch.tensor(input_ids)
prediction_masks = torch.tensor(attention_masks)
prediction_labels = torch.tensor(labels)
prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)


## Prediction on test set
# Put model in evaluation mode
model.eval()
# Tracking variables 
predictions , true_labels = [], []
# Predict 
i = 0
for batch in prediction_dataloader:
  print("step:", i)
  # Add batch to GPU
  batch = tuple(t.to(device) for t in batch)
  # Unpack the inputs from our dataloader
  b_input_ids, b_input_mask, b_labels = batch
  # Telling the model not to compute or store gradients, saving memory and speeding up prediction
  with torch.no_grad():
    # Forward pass, calculate logit predictions
    logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
  # Move logits and labels to CPU
  logits = logits.detach().cpu().numpy()
  label_ids = b_labels.to('cpu').numpy()  
  # Store predictions and true labels
  predictions.append(logits)
  true_labels.append(label_ids)
  i += 1
# Import and evaluate each test batch using Matthew's correlation coefficient
from sklearn.metrics import matthews_corrcoef
matthews_set = []
for i in range(len(true_labels)):
  matthews = matthews_corrcoef(true_labels[i],
                 np.argmax(predictions[i], axis=1).flatten())
  matthews_set.append(matthews)
  
# Flatten the predictions and true values for aggregate Matthew's evaluation on the whole dataset
flat_predictions = [item for sublist in predictions for item in sublist]
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
flat_true_labels = [item for sublist in true_labels for item in sublist]

print('Classification accuracy using BERT Fine Tuning: {0:0.2%}'.format(matthews_corrcoef(flat_true_labels, flat_predictions)))
