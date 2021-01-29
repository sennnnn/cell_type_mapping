# Cell2location Trying

## Data process

### Scanpy

Reference URL: https://cloud.tencent.com/developer/article/1610396

## Error

### Number #1

Problem Description
```
The program will be stuck to   
###Training model to determine n_epochs with CV###  
0%|                                           | 0/100 [00:00<?, ?it/s]  
```

Solve Method
```
This reason why this phenomenon will occer is that the dataloader is locked by multiple processes.  
Setting "num_workers=0" in the "fit_advi_iterative" function of "torch_model.py".
```

### Number #2

Problem Description
```
How to utilize GPU for pymc?
```

Solve Method
```
pymc3 is based on theano. So we need to configure GPU for theano. The configure tutorial can refer to https://blog.csdn.net/u011445467/article/details/108653211
```