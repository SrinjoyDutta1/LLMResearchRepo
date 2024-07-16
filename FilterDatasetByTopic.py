import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import numpy as np
from iblearn.over_sampling import SMOTE
import logging
import os


logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# Define the model save path
model_save_path = '/Users/srinjoydutta/Desktop/LLMResearch/model/bert_classification_model.pt'

# Load the training dataset
train_data_path = '/Users/srinjoydutta/Desktop/LLMResearch/dataset/SampledSimpleSelectTrainingDataset.csv'
train_df = pd.read_csv(train_data_path)

# Map categories to numbers
categories = [
    'Information security', 'ethics', 'thread safe questions', 
    'C pointer based questions', 'generic cs questions',
    'Development Tools and Practices', 'Programming Languages and Syntax', 
    'Software Design and Architecture', 'Database and SQL'
]
category_mapping = {idx: category for idx, category in enumerate(categories)}

# Adjust labels to be zero-indexed
train_df['Category'] = train_df['Category'].apply(lambda x: x - 1)

# Split into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['Body_question'], train_df['Category'], test_size=0.2, stratify=train_df['Category'], random_state=42
)

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the texts
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=256)
val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True, max_length=256)

# Convert data to arrays for SMOTE
train_inputs = np.array(train_encodings['input_ids'])
train_masks = np.array(train_encodings['attention_mask'])
train_labels = train_labels.values

# Apply SMOTE
smote = SMOTE(random_state=42)
train_inputs_res, train_labels_res = smote.fit_resample(train_inputs, train_labels)

# Resample train_masks to match train_inputs_res
train_masks_res, _ = smote.fit_resample(train_masks, train_labels)

# Convert back to tensors
train_inputs_res = torch.tensor(train_inputs_res)
train_masks_res = torch.tensor(train_masks_res)
train_labels_res = torch.tensor(train_labels_res)

# Convert validation data to tensors
val_inputs = torch.tensor(val_encodings['input_ids'])
val_masks = torch.tensor(val_encodings['attention_mask'])
val_labels = torch.tensor(val_labels.values)

train_dataset = TensorDataset(train_inputs_res, train_masks_res, train_labels_res)
val_dataset = TensorDataset(val_inputs, val_masks, val_labels)

train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=16)
val_loader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=16)

# Define the device for model and tensor computations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.array(list(range(len(categories)))), y=train_labels_res.numpy())
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Load pre-trained BERT model for classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(categories))

# Move model to device
model.to(device)

# Define optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * 5  # 5 epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Define loss function with class weights
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

# Training loop with early stopping
epochs = 5
best_accuracy = 0
early_stopping_tolerance = 2
early_stopping_counter = 0

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)
        optimizer.zero_grad()
        outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    avg_train_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f}')
    
    # Evaluate the model on the validation set
    model.eval()
    predictions, true_labels = [], []
    
    with torch.no_grad():
        for batch in val_loader:
            b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)
            outputs = model(b_input_ids, attention_mask=b_input_mask)
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=1).tolist())
            true_labels.extend(b_labels.tolist())
    
    accuracy = (np.array(predictions) == np.array(true_labels)).mean()
    print(f'Epoch {epoch + 1}/{epochs} | Validation Accuracy: {accuracy:.4f}')
    
    # Early stopping
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        early_stopping_counter = 0
        torch.save(model.state_dict(), model_save_path)
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_tolerance:
            print("Early stopping due to no improvement.")
            break

# Load the best model
model.load_state_dict(torch.load(model_save_path))

# Evaluate the best model on the validation set
model.eval()
predictions, true_labels = [], []

with torch.no_grad():
    for batch in val_loader:
        b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)
        outputs = model(b_input_ids, attention_mask=b_input_mask)
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, dim=1).tolist())
        true_labels.extend(b_labels.tolist())

# Include all possible class labels in the classification report
all_labels = list(range(len(categories)))
print(classification_report(true_labels, predictions, labels=all_labels, target_names=categories))

# Load the actual dataset to classify
actual_data_path = '/Users/srinjoydutta/Desktop/LLMResearch/dataset/RandomSampleOfFilteredQuestions.csv'
actual_df = pd.read_csv(actual_data_path)

# Tokenize the Body_question field in the actual dataset
actual_encodings = tokenizer(actual_df['Body_question'].tolist(), truncation=True, padding=True, max_length=256)

actual_inputs = torch.tensor(actual_encodings['input_ids'])
actual_masks = torch.tensor(actual_encodings['attention_mask'])

actual_dataset = TensorDataset(actual_inputs, actual_masks)
actual_loader = DataLoader(actual_dataset, batch_size=16, shuffle=False)

# Predict categories for the actual dataset
model.eval()
actual_predictions = []

with torch.no_grad():
    for batch in actual_loader:
        b_input_ids, b_input_mask = tuple(t.to(device) for t in batch)
        outputs = model(b_input_ids, attention_mask=b_input_mask)
        logits = outputs.logits
        actual_predictions.extend(torch.argmax(logits, dim=1).tolist())

# Map the predicted labels to categories
actual_df['Category'] = actual_predictions
actual_df['Category'] = actual_df['Category'].map(lambda x: category_mapping[x])

# Save the classified actual dataset
classified_data_path = 'LLMResearch/dataset/ClassifiedFilteredQuestions.csv'
actual_df.to_csv(classified_data_path, index=False)
