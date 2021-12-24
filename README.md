# Cross_Sample_Financial_Feature_Engineering

### Motivation
Let the neural network 'freely' learn the relationship between different samples. Here is an example in financial feature engineering. Because in the traditional neural network structure, if we have 100 features for stock1, we can't see the features of stock2 when training stock1. Of course, you can put the relationship of stock1 and stock2 into stock1 as a feature. However, you should know this relationship at first. In my framework, we can learn this cross-sample rules relationship **freely!!!** With the help of kernel, the neighborhood samples will get together at first. Although sample 1 maybe far away from sample 100, you may afraid that the 3*3 kernel maybe useless for them. But after using the kernel in several layers, the sample 1 and sample 100 will have a date, finally. 

Why not using GCN, this is a very straightward question. The good thing for GCN is that the operation will be more concise rely on the input adjacency matrix. If you don't input the adjacency matrix, in fact, from my opinion, GCN is a sub concept of CNNs. And I think CNN is more powerful. But if you design a good structure and have a good initialization knowledge, such as you know the relationship of different samples before training, there must be place for GCN. To sum up, I hope you know the right time to use the right model.

![Image text](https://github.com/Neural-Finance/Financial_Feature_Engineering_cross_sample/blob/master/fig/structure.PNG)


### Example
If we should rank() the price of stocks in the sample trading day. In order to let the neural network learn this operator, we have to let sample1 see the features belong to sample2. As metioned above, the traditional structure can't see the features belong to other sample. Thus, we should put all sample's data into one picture, and let it serves as X. The output should be all sample's value, which is Pred_Y. And the real factor value array should be Y. The mean squared error of Y and pred_Y, severs as loss function.


### Project Structure
```
Main.py --You can run it to train the network and test the data.
```

```
Data_processing.py --Built the figure data
```

```
Querry.py --a sub function for Data_processing, here, you can put in a formula, which produces x and y
```

```
Lenet.py --The neural network model structure (Tensorflow 1.3)
```

```
hyper_parameters.py --all the hyper-parameters
```

### Input X and Output Y
![Image text](https://github.com/Neural-Finance/Cross_sample_financial_feature_engineering/blob/master/fig/1.png)
![Image text](https://github.com/Neural-Finance/Cross_sample_financial_feature_engineering/blob/master/fig/2.png)

**Lenet (A kind of CNN network structure)**

I want to mention that, sometimes, 1*m kernels will be more helpful in this task, because it's more like a time series operation. And the second point is that, I know GNN is very hot nowadays, however, you should know the relationship between different stocks. If you are pretty sure about it, then you should use GNN here. If you want to freely learn their relationship, both in time series or cross section relationship, then the tradtional CNN will be a better choice.

![Image text](https://github.com/Neural-Finance/Cross_sample_financial_feature_engineering/blob/master/fig/5.png)

### Experiment Result

**Valid Gradient Descent**

![Image text](https://github.com/Neural-Finance/Cross_sample_financial_feature_engineering/blob/master/fig/3.png)

**Prediction and Real**

![Image text](https://github.com/Neural-Finance/Cross_sample_financial_feature_engineering/blob/master/fig/4.png)

This result can be much better if you train it with GPU. I show the result on my laptop, thus, it hasn't got enough training. However, as you can see, our method sucessfully learns the rank() operator. This framework can help you learn the relationship between different stocks. **If it's helpful, please give me a star, thanks.**
