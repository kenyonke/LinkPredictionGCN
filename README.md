# Learning Edge Features based on GCN for Link Prediction

Model Design:
<img src='https://github.com/kenyonke/LinkPredictionGCN/blob/master/model.JPG'>

## Requierments
```
tensorflow(>0.12)
networkx
```


## Preprocess the Data
```
cd LPG
python RemoveLinks.py (generate the training link and testing links for the model)
```

## Run the code:
```
cd LPG
python train.py
```

## Reference
```
@inproceedings{kipf2017semi,
  title={Semi-Supervised Classification with Graph Convolutional Networks},
  author={Kipf, Thomas N. and Welling, Max},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2017}
}
```
