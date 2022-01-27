import pandas as pd
import numpy as np
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers_interpret import SequenceClassificationExplainer
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import torch
from sklearn.metrics import f1_score
import pickle
from collections import Counter
from transformers import logging
logging.set_verbosity_error()
import json
import csv


def get_word_attributions(model, tokenizer, data, removed):
    '''
    given as input the pretrained model, the tokenizer and a set of text data, this function make explanations for each articles and outputs 
    a dict with as key the predicted class and as value - for each article belonging to that class- a list of tuple(word, word_attribution) saying how much that word 
    has contributed to the predicted class.

    'model': pretrained model,
    'tokenizer': pretrained tokenizer,
    'data': dataframe of texts to be explained
    '''

    # model classes
    #print('Classes:', model.config.id2label)
    
    multiclass_explainer = SequenceClassificationExplainer(
        model,
        tokenizer)
        
    # list of articles to explain
    texts = data.short_maintext.to_list()
    src = data.source_name.to_list()
    #src_names = list(set(data.source_name.to_list()))
    real_class = data.political_leaning.to_list()
    mapping = {'LABEL_1':'Liberal','LABEL_0':'Conservative', 'LABEL_2':'Centre-left', 'LABEL_4':'Left-wing', 'LABEL_3':'Liberal-conservative','LABEL_5':'Right-wing'}
    # mapping src_name:political_label
    with open(r'/homenfs/l.bellomo1/datasets/new_attempt_classifier/explainability/mappings/media_label.json', encoding='utf-8') as fp:
        media_label = json.loads(fp.read())
    src_names = list(media_label.keys())
    corrected_pred = 0
    '''
    Compute explanations
    '''
    # dict to store class-explanations
    class_explained = dict() 

    # for each article in the test set compute the explanation and then putting it into a list res
    for i in range(len(texts)): 
        # select only the first 512 tokens as in the training set
        text = texts[i][:512] 
        #print(text)
        
        # word attributions for each article
      
        word_attributions = multiclass_explainer(text)

        if real_class[i] == mapping[multiclass_explainer.predicted_class_name]:
            corrected_pred +=1
        #print(corrected_pred)
        
        if removed: 
            multiclass_explainer.visualize(f"/homenfs/l.bellomo1/datasets/new_attempt_classifier/explainability/graph/media_name_pattern/mBERT/without_medianame_local_expl.html")
        else:
            multiclass_explainer.visualize(f"/homenfs/l.bellomo1/datasets/new_attempt_classifier/explainability/graph/media_name_pattern/mBERT/with_medianame_local_expl.html")

    print('accuracy', corrected_pred/len(texts))
    return corrected_pred/len(texts)



'''
get_word_attributions() Input
'''
'''
mBERT model
'''
model_path = r'/homenfs/l.bellomo1/datasets/new_attempt_classifier/explainability/models/MultiBert_6class_all/punt/finetuned_MBERT6class_epoch_3_new.model'
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased",
                                                      num_labels=6,
                                                      output_attentions=False,
                                                      output_hidden_states=False)

model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
# tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', 
                                          do_lower_case=False)
'''
XLM model

# model
model_path = r'/homenfs/l.bellomo1/datasets/new_attempt_classifier/explainability/models/XLM/punt/finetuned_XLM_epoch_3.model'
model = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base",
                                                      num_labels=6,
                                                      output_attentions=False,
                                                      output_hidden_states=False)
model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
# tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base', 
                                          do_lower_case=False)
'''
'''
 data to explain: picking all media that cite theirself in upper characters inside the article
'''
# src_names contained in test biased
test_bias_punt = pd.read_csv(r'/homenfs/l.bellomo1/datasets/new_attempt_classifier/explainability/data_to_explain/test_bias_cleaned.csv')
# shortening the legth of sequence as for bert tokenizer
test_bias_punt['short_maintext'] = test_bias_punt['maintext'].str.slice(0,600)
print(len(test_bias_punt))
# src_names contained in test biased
src_names = list(test_bias_punt.source_name.unique())
src_names = [elem.upper() for elem in src_names]
test_contains = test_bias_punt[test_bias_punt.short_maintext.str.contains('|'.join(src_names),case=True)]
print(len(test_contains.source_name.unique()))
print(len(test_contains))

#print(src_names[:3])
#exit()
acc_with_media = get_word_attributions(model, tokenizer, test_contains, removed=False)
print('acc_with_media', acc_with_media)
'''
 data to explain: removing all media name that cite theirself in upper characters inside the article
'''
# src_names contained in test biased
test_bias_punt = pd.read_csv(r'/homenfs/l.bellomo1/datasets/new_attempt_classifier/explainability/data_to_explain/test_bias_cleaned.csv')
# shortening the legth of sequence as for bert tokenizer
test_bias_punt['short_maintext'] = test_bias_punt['maintext'].str.slice(0,600)
print(len(test_bias_punt))
# src_names contained in test biased
src_names = list(test_bias_punt.source_name.unique())
src_names = [elem.upper() for elem in src_names]
test_contains = test_bias_punt[test_bias_punt.short_maintext.str.contains('|'.join(src_names),case=True)]
test_contains['short_maintext'] = test_contains['short_maintext'].str.replace('|'.join(src_names),'')
print(len(test_contains))

acc_without_media = get_word_attributions(model, tokenizer, test_contains, removed=True)
print('acc_without_media', acc_without_media)

'''
compute delta accuracy
'''
delta_acc = acc_with_media - acc_without_media
print('delta_acc', delta_acc)

