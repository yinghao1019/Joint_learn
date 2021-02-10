# Joint_learn  
Pytorch implementation of below Model  
   *`JointBert`:[BERT for Joint Intent Classification and Slot Filling](https://arxiv.org/abs/1902.10909)  
   *`AttnSeq2Seq`:[Attention-Based Recurrent Neural Network Models for Joint Intent Detection and Slot Filling](https://arxiv.org/abs/1609.01454)  
  
## Model Architecture  
  
#### 1.JoinBert
Architecture is referenced to [monologg/JointBERT](https://github.com/monologg/JointBERT).Please click to see detail.   
    - Predict `intent` and `slot` at the same time from **one BERT model** (=Joint model)  
    - total_loss = intent_loss + coef \* slot_loss (Change coef with `--slot_loss_coef` option)  
    - **If you want to use CRF layer, give `--use_crf` option**  
  
#### 2.AttnSeq2Seq  
  
      ![]()

    Using Attention mechanism based on RNN Encoder-Decoder.  
    -Encoder part  
     1. Bidirectional Rnn(LSTM) to encode source sents.
     2. Backward Lstm final hidden state to compute deocder init hidden state.  
      
    -Attenion part
     1.Using neural network,last hidden state,encoder hiddens to compute  attention weight.
     2.using softmax to gain wieght  
     
    -Decoder part
    1.To predict current slots,feed last hidden state,last predict label,aligned encoder hidden,
    context vector.  
    2.using last hidden state & encoder hiddens to compute current context vector  
    3.using init decoder hidden & it's context to compute intent classification  
    4.total_loss= intent_loss + coef \* slot_loss  
  
## Dependencies  
- python>=3.7
- torch==
- seqeval==
- transformers==
- pytorch-crf==
  
## Dataset
|       | Train  | Dev | Test | Intent Labels | Slot Labels |
| ----- | ------ | --- | ---- | ------------- | ----------- |
| ATIS  | 4,478  | 500 | 893  | 21            | 120         |
| Snips | 13,084 | 700 | 700  | 7             | 72          |

- The number of labels are based on the _train_ dataset.
- Add `UNK` for labels (For intent and slot labels which are only shown in _dev_ and _test_ dataset)
- Add `PAD` for slot label

## Train & Evaluation  
  
```bash  
$ python main.py --

#For ATIS

#For Snips
```  
  
## Prediction  
  
```bash  
$ python main.py --

#For ATIS

#For Snips
```  
  
## Results

- Run 5 ~ 10 epochs (Record the best result)
- Only test with `uncased` model
- Warm up steps 248 is the best

|           |                  | Intent acc (%) | Slot F1 (%) | Sentence acc (%) |
| --------- | ---------------- | -------------- | ----------- | ---------------- |
| **Snips** | BERT             | **99.14**      | 96.90       | 93.00            |
|           | BERT + CRF       | 98.57          | **97.24**   | **93.57**        |
|           | DistilBERT       | 98.00          | 96.10       | 91.00            |
|           | DistilBERT + CRF | 98.57          | 96.46       | 91.85            |
|           | ALBERT           | 98.43          | 97.16       | 93.29            |
|           | ALBERT + CRF     | 99.00          | 96.55       | 92.57            |
| **ATIS**  | BERT             | 97.87          | 95.59       | 88.24            |
|           | BERT + CRF       | **97.98**      | 95.93       | 88.58            |
|           | DistilBERT       | 97.76          | 95.50       | 87.68            |
|           | DistilBERT + CRF | 97.65          | 95.89       | 88.24            |
|           | ALBERT           | 97.64          | 95.78       | 88.13            |
|           | ALBERT + CRF     | 97.42          | **96.32**   | **88.69**        |

## Sentence predict Result  

## References  
  
- [Huggingface Transformers](https://github.com/huggingface/transformers)  
- [pytorch-crf](https://github.com/kmkurn/pytorch-crf)  
- [monologg/JointBERT](https://github.com/monologg/JointBERT)