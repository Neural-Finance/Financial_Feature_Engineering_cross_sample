# Cross_Sample_Financial_Feature_Engineering

### Motivation
Let the neural network 'freely' learn the relationship between different samples. Here is an example in financial feature engineering. Because in the traditional neural network structure, if we have 100 features for stock1, we can't see the features of stock2 when training stock1. Of course, you can put the relationship of stock1 and stock2 into stock1 as a feature. However, you should know this relationship at first. In my framework, we can learn this relationship **freely!!!** 

### Example
Here is a financial factor, we should rank() the price of stocks in the sample trading day. In order to let the neural network learn this operator, we have to let sample1 see the features belong to sample2.


### Network Structure
--------------------------------------

Main.py --You can run it to train the network and test the data.

--------------------------------------

Data_processing.py --Built the figure data

--------------------------------------

Querry.py --a sub function for Data_processing, here, you can put in a formula, which produces x and y

--------------------------------------

Lenet.py --The neural network model structure

--------------------------------------

hyper_parameters.py --all the hyper-parameters

--------------------------------------

**Lenet (A kind of CNN network structure)**
![Image text](https://github.com/Neural-Finance/Cross_sample_financial_feature_engineering/blob/master/fig/5.png)

### Input X and Output Y
![Image text](https://github.com/Neural-Finance/Cross_sample_financial_feature_engineering/blob/master/fig/1.png)
![Image text](https://github.com/Neural-Finance/Cross_sample_financial_feature_engineering/blob/master/fig/2.png)

### Experiment Result
**Valid Gradient Descent**
![Image text](https://github.com/Neural-Finance/Cross_sample_financial_feature_engineering/blob/master/fig/3.png)
**Prediction and Real**
![Image text](https://github.com/Neural-Finance/Cross_sample_financial_feature_engineering/blob/master/fig/4.png)
This result can be much better if you train it with GPU. I show the result on my laptop, thus, it hasn't got enough training. However, as you can see, our method sucessfully learns the rank() operator. This framework can help you learn the relationship between different stocks. **If it's helpful, please give me a star, thanks.**
