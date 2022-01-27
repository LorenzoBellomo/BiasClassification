# importing libraries
import os
os.nice(10)
import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
import random
import matplotlib.pyplot as plt
import csv


'''
    Compute Performance Metrics (F1 Score, Accuracy)
'''
def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')

def accuracy_per_class(preds, labels, label_dict):
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')

''' 
    Evaluation
'''

def evaluate(dataloader_val, device, model, _field, epoch, model_type, _type):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_val:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                'attention_mask': batch[1],
                'labels':         batch[2],
                }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    int_pred = np.argmax(predictions, axis=1)
    true_vals = np.concatenate(true_vals, axis=0)
    
    rows = zip(int_pred,true_vals)
    with open(f'/homenfs/l.bellomo1/datasets/new_attempt_classifier/explainability/models/{model_type}/{_field}/{_type}/true_pred_class_epoch_{epoch}', "w") as f: # csv file with predictions (pred_class, true_class)
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

        
    return loss_val_avg, predictions, true_vals

def train_model(df, model_type, _type, _field):
    ''' 
        Data Preparation
    '''

    assigned_labels = {'Liberal': 1, 'Conservative': 0, 'Centre-left': 2, 'Left-wing': 4, 'Liberal-conservative': 3, 'Right-wing': 5}
    df['label'] = df['political_leaning'].map(assigned_labels)
    print('len Dataset 6-class:', len(df)) # TODO: prova meno classi
    # dropping unecessary columns for classification
    df.drop(columns=['date_publish','language','source_domain','id','country','source_name', 'year'], inplace=True) # TODO: prova mergendo maintext con title e solo title
    print('n_articles for each label:', df['political_leaning'].value_counts())
    # defining Training and Validation Split
    X_train, X_val, y_train, y_val = train_test_split(df.index.values, 
                                                    df.label.values, 
                                                    test_size=0.10, 
                                                    random_state=42, 
                                                    stratify=df.label.values)

    df['data_type'] = ['not_set']*df.shape[0]
    df.loc[X_train, 'data_type'] = 'train'
    df.loc[X_val, 'data_type'] = 'val'
    print(df.groupby(['label', 'data_type']).count())
    # label dict 
    label_dict = pd.Series(df.label.values,index=df.political_leaning).to_dict()
    print(label_dict)

    '''
        Tokenizing and Encoding Data
    '''
    # BertTokenizer: tokenizing texts and turning them into integers vectors
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', 
                                            do_lower_case=False) # TODO: prova anche con True
                                            
    encoded_data_train = tokenizer.batch_encode_plus(
        df[df.data_type=='train'][_field].values, 
        add_special_tokens=True, 
        return_attention_mask=True, 
        padding=True,
        truncation=True, 
        return_tensors='pt'
    )
    print('end tokenization training')
    encoded_data_val = tokenizer.batch_encode_plus(
        df[df.data_type=='val'][_field].values, 
        add_special_tokens=True, 
        return_attention_mask=True, 
        padding=True, 
        truncation=True, 
        return_tensors='pt'
    )
    print('end tokenization validation')
    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']
    labels_train = torch.tensor(df[df.data_type=='train'].label.values)

    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val['attention_mask']
    print(type(df[df.data_type=='val'].label.values))
    labels_val = torch.tensor(df[df.data_type=='val'].label.values)

    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)
    print('end tokenization and encoding')

    '''
    Uploading BERT Model and defining parameters
    '''
    # BERT Pre-trained Model: bert-base-uncased is a smaller pre-trained model
    model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased",
                                                        num_labels=len(label_dict),
                                                        output_attentions=False,
                                                        output_hidden_states=False)

    # Data Loaders: DataLoader combines a dataset and a sampler, and provides an iterable over the given dataset
    batch_size = 2
    dataloader_train = DataLoader(dataset_train, 
                                sampler=RandomSampler(dataset_train), 
                                batch_size=batch_size)
    dataloader_validation = DataLoader(dataset_val, 
                                    sampler=SequentialSampler(dataset_val), 
                                    batch_size=batch_size)

    optimizer = AdamW(model.parameters(),
                    lr=1e-5, 
                    eps=1e-8)                 
    epochs = 3
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=0,
                                                num_training_steps=len(dataloader_train)*epochs)


    '''
        Training Loop
    '''
    seed_val = 17
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    training_stats = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(device)

        
    for epoch in range(1, epochs+1):
        
        model.train()
        
        loss_train_total = 0

        for batch in dataloader_train:

            model.zero_grad()
            
            batch = tuple(b.to(device) for b in batch)
            
            inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'labels':         batch[2],
                    }       

            outputs = model(**inputs)
            
            loss = outputs[0]
            loss_train_total += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            
        # TODO change path
        torch.save(model.state_dict(), f'/homenfs/l.bellomo1/datasets/new_attempt_classifier/explainability/models/{model_type}/{_field}/{_type}/finetuned_MBERT6class_epoch_{epoch}.model')
        # recording performances from this epoch     
        print(f'\nEpoch {epoch}')
        loss_train_avg = loss_train_total/len(dataloader_train)            
        print(f'Training loss: {loss_train_avg}')
        val_loss, predictions, true_vals = evaluate(dataloader_validation, device, model, _field, epoch, model_type, _type)
        val_f1 = f1_score_func(predictions, true_vals)
        val_accuracy = accuracy_per_class(predictions, true_vals, label_dict)
        print(f'Validation loss: {val_loss}')
        print(f'F1 Score (Weighted): {val_f1}')
        training_stats.append(
            {
                'epoch': epoch,
                'Training Loss': loss_train_avg,
                'Valid. Loss': val_loss,
                'Vaid. F1 Score': val_f1,
            }
        )

    '''
        Saving and plotting performances across all epochs
    '''
    # creating a DataFrame from our training statistics.
    df_stats = pd.DataFrame(data=training_stats)
    df_stats = df_stats.set_index('epoch')
    print(df_stats)
    df_stats.to_csv(f'/homenfs/l.bellomo1/datasets/new_attempt_classifier/explainability/models/{model_type}/{_field}/{_type}/MBERT_training_stats.csv')    # plot Learning Curve
    plt.rcParams["figure.figsize"] = (12,6)
    plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
    plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")
    plt.title("mBERT - Learning Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks([1, 2, 3])
    plt.savefig(f'/homenfs/l.bellomo1/datasets/new_attempt_classifier/explainability/models/{model_type}/{_field}/{_type}/MBERT_learningcurve.png') 
    plt.show()

'''
 main
'''

'''
    maintext
'''

### mBERT maintext training punt
print('\n')
print('----------------------------------')
print('mBERT maintext training punt')
print('----------------------------------')
print('\n')
df = pd.read_csv('/homenfs/l.bellomo1/datasets/new_attempt_classifier/datasets/training_punt_cleaned.csv')
_type = 'punt'
model_type = 'MultiBert_6class_all'
_field = 'maintext' 
train_model(df, model_type, _type, _field)

### mBERT maintext training no punt
print('\n')
print('----------------------------------')
print('mBERT maintext training no punt')
print('----------------------------------')
print('\n')
df = pd.read_csv('/homenfs/l.bellomo1/datasets/new_attempt_classifier/datasets/training_cleaned_nopunt.csv')
_type = 'no_punt'
model_type = 'MultiBert_6class_all'
_field = 'maintext'
train_model(df, model_type, _type, _field)

'''
    title
'''
### mBERT title training punt
print('\n')
print('----------------------------------')
print('mBERT title training punt')
print('----------------------------------')
print('\n')
df = pd.read_csv('/homenfs/l.bellomo1/datasets/new_attempt_classifier/datasets/training_punt_cleaned.csv')
#df = df.groupby('political_leaning').apply(lambda x: x.sample(n=10))
_type = 'punt'
model_type = 'MultiBert_6class_all'
_field = 'title' 
train_model(df, model_type, _type, _field)

### mBERT title training no punt
print('\n')
print('----------------------------------')
print('mBERT title training no punt')
print('----------------------------------')
print('\n')
df = pd.read_csv('/homenfs/l.bellomo1/datasets/new_attempt_classifier/datasets/training_cleaned_nopunt.csv')
#df = df.groupby('political_leaning').apply(lambda x: x.sample(n=10))
_type = 'no_punt'
model_type = 'MultiBert_6class_all'
_field = 'title'
train_model(df, model_type, _type, _field)


'''
    all_text (title + maintext)
'''
### mBERT title training punt
print('\n')
print('----------------------------------')
print('mBERT all_text (title + maintext) training punt')
print('----------------------------------')
print('\n')
df = pd.read_csv('/homenfs/l.bellomo1/datasets/new_attempt_classifier/datasets/training_punt_cleaned.csv')
#df = df.groupby('political_leaning').apply(lambda x: x.sample(n=10))
_type = 'punt'
model_type = 'MultiBert_6class_all'
_field = 'all_text' 
train_model(df, model_type, _type, _field)

### mBERT title training no punt
print('\n')
print('----------------------------------')
print('mBERT all_text (title + maintext)  training no punt')
print('----------------------------------')
print('\n')
df = pd.read_csv('/homenfs/l.bellomo1/datasets/new_attempt_classifier/datasets/training_cleaned_nopunt.csv')
#df = df.groupby('political_leaning').apply(lambda x: x.sample(n=10))
_type = 'no_punt'
model_type = 'MultiBert_6class_all'
_field = 'all_text'
train_model(df, model_type, _type, _field)
