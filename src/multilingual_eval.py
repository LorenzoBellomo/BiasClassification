import os
os.nice(10)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import f1_score
import pickle
from collections import Counter
import seaborn as sns
from transformers import logging
logging.set_verbosity_error()
import csv
torch.multiprocessing.set_sharing_strategy('file_system')


'''
Compute Performances Metrics
'''
def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')
def accuracy_per_class(preds, labels, label_dict):
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    
    #preds_flat = np.argmax(preds, axis=1).flatten()
    preds_flat = np.argmax(preds, axis=1)
    labels_flat = labels
    correct_idx = (preds_flat == labels_flat)

    for label in np.unique(labels_flat):
        #y_preds = preds_flat[labels_flat==label]
        #y_true = labels_flat[labels_flat==label]
        n_correct = np.sum(labels_flat[correct_idx] == label)
        n_label = np.sum(labels_flat == label)
        
        print(f'Class: {label} {label_dict_inverse[label]}, ')
        print(f'{n_correct}/{len(labels_flat[labels_flat==label])}', n_correct/len(labels_flat[labels_flat==label]) )
        #print((f'{len(y_preds[y_preds==label])}/{len(y_true)}', len(y_preds[y_preds==label])/len(y_true) ))
        #print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')
        #print('Accuracy %:',len(y_preds[y_preds==label])/len(y_true))



'''
Making Predictions on unseen data
'''

def evaluate(dataloader_val, model, device):

    model.eval()
    
    predictions, true_vals = [], []
    for batch in dataloader_val:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                'attention_mask': batch[1],
                'labels':         batch[2],
                }

        with torch.no_grad():        
            outputs = model(**inputs)

        logits = outputs[1]

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
    int_pred = np.argmax(predictions, axis=1)
    rows = zip(int_pred,true_vals)
    
    return  predictions, true_vals, rows

'''
    Plot Confusion Matrix
'''    

def Plot_ConfusionMatrix(true, pred, model_type, _type, biased):
  n_rows, n_cols = 1, 1
  f, ax= plt.subplots(n_rows, n_cols, figsize=(8,5))
  sns.heatmap(confusion_matrix(true, pred), annot=True, fmt='g', ax=ax)  #annot=True to annotate cells, ftm='g' to disable scientific notation

  # set labels
  true = set(true)
  label_dict = {4: 'Left-wing', 0: 'Conservative', 3: 'Liberal-conservative', 5: 'Right-wing', 1: 'Liberal', 2: 'Centre-left'}
  label = [label_dict[elem] for elem in true]
  
  # labels, title and ticks
  ax.set_xlabel('True labels', fontsize=20)
  ax.set_ylabel('Predicted labels', fontsize=20)
  ax.xaxis.set_ticklabels(label, rotation=90)
  ax.yaxis.set_ticklabels(label, rotation=0)
  plt.tight_layout()
  if biased:
    plt.savefig(f'/homenfs/l.bellomo1/datasets/new_attempt_classifier/explainability/models/{model_type}/{_type}/confusion_matrix_test_bias.jpg')
  else:
      plt.savefig(f'/homenfs/l.bellomo1/datasets/new_attempt_classifier/explainability/models/{model_type}/{_type}/confusion_matrix_test_no_bias.jpg')



'''
 main function to make test set evaluation
'''

def TestSet_Evaluation(df, model_path, _type, model_type, biased):
    # setting device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device ='cpu'
    print('Device',device)
    '''
    Data Preparation
    '''
    # dropping unecessary columns for classification
    #df.drop(columns=['title','date_publish','language','source_domain','id','country','source_name', 'year'], inplace=True) # TODO: prova mergendo maintext con title e solo title
    df.drop(columns=['title','date_publish','language','source_domain','source_name', 'year'], inplace=True) # TODO: prova mergendo maintext con title e solo title
    print('n_articles for each label:', df['political_leaning'].value_counts())
    #print(df.groupby(['label', 'data_type']).count())
    # label dict 
    #label_dict = pd.Series(df.label.values,index=df.political_leaning).to_dict()
    label_dict = {'Left-wing': 4, 'Conservative': 0, 'Liberal-conservative': 3, 'Right-wing': 5, 'Liberal': 1, 'Centre-left': 2}
    print(label_dict)
    
    # BertTokenizer: tokenizing texts and turning them into integers vectors
    if model_type == 'MultiBert_6class_all':
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', 
                                          do_lower_case=False)
    elif model_type == 'XLM':
        tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base', 
                                          do_lower_case=False) # TODO: prova anche con True
    # encoding tokenized texts to indexes
    encoded_data_val = tokenizer.batch_encode_plus(
    df.maintext.values, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    padding=True, 
    truncation=True, 
    return_tensors='pt'
    )
    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val['attention_mask']
    labels_val = torch.tensor(df.label.values)
    dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)
    batch_size = 2
    # iterable DatLoader
    dataloader_testset = DataLoader(dataset_val, 
                                   sampler=SequentialSampler(dataset_val), 
                                   batch_size=batch_size)
    '''
    Loading and Evaluating Model
    '''
    if model_type == 'MultiBert_6class_all':
        model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased",
                                                      num_labels=len(label_dict),
                                                      output_attentions=False, #TODO: rimetti True quando vuoi calcolare attention
                                                      output_hidden_states=False)
    elif model_type == 'XLM':
        model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base',
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,  #TODO: rimetti True quando vuoi calcolare attention
                                                      output_hidden_states=False)                                           
    model.to(device)
    #model.load_state_dict(torch.load(model_path))

    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    #predictions, true_vals, rows, input_ids, attention = evaluate(dataloader_testset, model, device)  #TODO: rimetti  quando vuoi calcolare attention
    #predictions, true_vals, rows, input_ids = evaluate(dataloader_testset, model, device)
    predictions, true_vals, rows = evaluate(dataloader_testset, model, device)
    #batch_sents = np.concatenate(input_ids, axis=0)  #TODO: rimetti  quando vuoi calcolare attention
    #converted = [tokenizer.convert_ids_to_tokens(sent) for sent in batch_sents] #TODO: rimetti  quando vuoi calcolare attention
    # return converted, attention (to extract attention and converted ids to token)
    
    if biased == False:
        with open(f'/homenfs/l.bellomo1/datasets/new_attempt_classifier/explainability/models/{model_type}/{_type}/testset_pred/true_pred_class_6class_nobias', "w") as f: # csv file with predictions (pred_class, true_class)
            writer = csv.writer(f)
            for row in rows:
                writer.writerow(row)
    else:
        with open(f'/homenfs/l.bellomo1/datasets/new_attempt_classifier/explainability/models/{model_type}/{_type}/testset_pred/true_pred_class_6class_bias', "w") as f: # csv file with predictions (pred_class, true_class)
            writer = csv.writer(f)
            for row in rows:
                writer.writerow(row)
   

    # Computing Test Set performances
    print('\n')
    val_f1 = f1_score_func(predictions, true_vals)
    print(f'F1 Score (Weighted): {round(val_f1,3)}')
    print('\n')
    print('Accuracy:')
    val_accuracy = accuracy_per_class(predictions, true_vals, label_dict)

    if biased == False:
        pred_true_vals = pd.read_csv(f'/homenfs/l.bellomo1/datasets/new_attempt_classifier/explainability/models/{model_type}/{_type}/testset_pred/true_pred_class_6class_nobias', names=['pred_class','true_class'])
    else:
        pred_true_vals = pd.read_csv(f'/homenfs/l.bellomo1/datasets/new_attempt_classifier/explainability/models/{model_type}/{_type}/testset_pred/true_pred_class_6class_bias', names=['pred_class','true_class'])
    
    pred = pred_true_vals['pred_class'].to_list()
    true = pred_true_vals['true_class'].to_list()
    df['pred_class'] = pred
    df['true_class'] = true
    Plot_ConfusionMatrix(true, pred, model_type, _type, biased)



def call_test_set(df, model_path, model_type, _type, biased):

    # TODO rimetti if
    #if biased == True:
        #df = df.groupby('political_leaning').apply(lambda x: x.sample(n=7500))
    assigned_labels = {'Liberal': 1, 'Conservative': 0, 'Centre-left': 2, 'Left-wing': 4, 'Liberal-conservative': 3, 'Right-wing': 5}
    df['label'] = df['political_leaning'].map(assigned_labels)
    # Report the number of sentences.
    print('Number of test sentences: {:,}\n'.format(df.shape[0]))
    TestSet_Evaluation(df, model_path, _type, model_type, biased)

'''
 main
'''

### mBERT punt test bias
print('\n')
print('----------------------------------')
print('mBERT punt test bias')
print('----------------------------------')
print('\n')
df = pd.read_csv('/homenfs/l.bellomo1/datasets/new_attempt_classifier/datasets/test_bias_cleaned.csv')
df = df.groupby('political_leaning').apply(lambda x: x.sample(n=7500))
_type = 'punt'
biased = True
model_type = 'MultiBert_6class_all'
model_path = f'/homenfs/l.bellomo1/datasets/new_attempt_classifier/explainability/models/MultiBert_6class_all/punt/finetuned_MBERT6class_epoch_3_new.model'
call_test_set(df, model_path, model_type, _type, biased)


### mBERT punt test no bias
print('\n')
print('----------------------------------')
print('mBERT punt test no bias')
print('----------------------------------')
print('\n')
df = pd.read_csv("/homenfs/l.bellomo1/datasets/new_attempt_classifier/datasets/test_no_bias_balanced_cleaned.csv")
_type = 'punt'
biased = False
model_type = 'MultiBert_6class_all'
model_path = f'/homenfs/l.bellomo1/datasets/new_attempt_classifier/explainability/models/MultiBert_6class_all/punt/finetuned_MBERT6class_epoch_3_new.model'
call_test_set(df, model_path, model_type, _type, biased)

### mBERT no-punt test bias
print('\n')
print('----------------------------------')
print('mBERT no-punt test bias')
print('----------------------------------')
print('\n')
df = pd.read_csv("/homenfs/l.bellomo1/datasets/new_attempt_classifier/datasets/test_bias_cleaned_nopunt.csv")
df = df.groupby('political_leaning').apply(lambda x: x.sample(n=7500))
_type = 'no_punt'
biased = True
model_type = 'MultiBert_6class_all'
model_path = f'/homenfs/l.bellomo1/datasets/new_attempt_classifier/explainability/models/MultiBert_6class_all/no_punt/finetuned_MBERT6class_epoch_2_new.model'
call_test_set(df, model_path, model_type, _type, biased)

### mBERT no-punt test no bias
print('\n')
print('----------------------------------')
print('mBERT no-punt test no bias')
print('----------------------------------')
print('\n')
df = pd.read_csv('/homenfs/l.bellomo1/datasets/new_attempt_classifier/datasets/test_no_bias_balanced_cleaned_nopunt.csv')
_type = 'no_punt'
biased = False
model_type = 'MultiBert_6class_all'
model_path = f'/homenfs/l.bellomo1/datasets/new_attempt_classifier/explainability/models/MultiBert_6class_all/no_punt/finetuned_MBERT6class_epoch_2_new.model'
call_test_set(df, model_path, model_type, _type, biased)

### XLM punt test bias
print('\n')
print('----------------------------------')
print('XLM punt test bias')
print('----------------------------------')
print('\n')
df = pd.read_csv('/homenfs/l.bellomo1/datasets/new_attempt_classifier/datasets/test_bias_cleaned.csv')
df = df.groupby('political_leaning').apply(lambda x: x.sample(n=7500))
_type = 'punt'
biased = True
model_type = 'XLM'
model_path = f'/homenfs/l.bellomo1/datasets/new_attempt_classifier/explainability/models/XLM/punt/finetuned_XLM_epoch_3.model'
call_test_set(df, model_path, model_type, _type, biased)

### XLM punt test no bias
print('\n')
print('----------------------------------')
print('XLM punt test no bias')
print('----------------------------------')
print('\n')
df = pd.read_csv("/homenfs/l.bellomo1/datasets/new_attempt_classifier/datasets/test_no_bias_balanced_cleaned.csv")
_type = 'punt'
biased = False
model_type = 'XLM'
model_path = f'/homenfs/l.bellomo1/datasets/new_attempt_classifier/explainability/models/XLM/punt/finetuned_XLM_epoch_3.model'
call_test_set(df, model_path, model_type, _type, biased)

### XLM no-punt test bias
print('\n')
print('----------------------------------')
print('XLM no-punt test bias')
print('----------------------------------')
print('\n')
df = pd.read_csv("/homenfs/l.bellomo1/datasets/new_attempt_classifier/datasets/test_bias_cleaned_nopunt.csv")
df = df.groupby('political_leaning').apply(lambda x: x.sample(n=7500))
_type = 'no_punt'
biased = True
model_type = 'XLM'
model_path = f'/homenfs/l.bellomo1/datasets/new_attempt_classifier/explainability/models/XLM/no_punt/finetuned_XLM_epoch_3.model'
call_test_set(df, model_path, model_type, _type, biased)


### XLM no-punt test no bias
print('\n')
print('----------------------------------')
print('XLM no-punt test no bias')
print('----------------------------------')
print('\n')
df = pd.read_csv('/homenfs/l.bellomo1/datasets/new_attempt_classifier/datasets/test_no_bias_balanced_cleaned_nopunt.csv')
_type = 'no_punt'
biased = False
model_type = 'XLM'
model_path = f'/homenfs/l.bellomo1/datasets/new_attempt_classifier/explainability/models/XLM/no_punt/finetuned_XLM_epoch_3.model'
call_test_set(df, model_path, model_type, _type, biased)