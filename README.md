
# Bias Variance Tradeoff + More Overfitting

When modelling, we are trying to create a useful prediction that can help us in the future. When doing this, we have seen how we need to create a train test split in order to keep ourselves honest in tuning our model to the data itself. Another perspective on this problem of overfitting versus underfitting is the bias variance tradeoff. We can decompose the mean squared error of our models in terms of bias and variance to further investigate.

$ E[(y-\hat{f}(x)^2] = Bias(\hat{f}(x))^2 + Var(\hat{f}(x)) + \sigma^2$
  
  
$Bias(\hat{f}(x)) = E[\hat{f}(x)-f(x)]$  
$Var(\hat{f}(x)) = E[\hat{f}(x)^2] - \big(E[\hat{f}(x)]\big)^2$

<img src="./images/bias_variance.png" alt="Drawing" style="width: 500px;"/>

## 1. Split the data into a test and train set.


```python
import pandas as pd
df = pd.read_excel('./movie_data_detailed_with_ols.xlsx')
def norm(col):
    minimum = col.min()
    maximum = col.max()
    return (col-minimum)/(maximum-minimum)
for col in df:
    try:
        df[col] = norm(df[col])
    except:
        pass
X = df[['budget','imdbRating','Metascore','imdbVotes']]
y = df['domgross']
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>budget</th>
      <th>domgross</th>
      <th>title</th>
      <th>Response_Json</th>
      <th>Year</th>
      <th>imdbRating</th>
      <th>Metascore</th>
      <th>imdbVotes</th>
      <th>Model</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.034169</td>
      <td>0.055325</td>
      <td>21 &amp;amp; Over</td>
      <td>NaN</td>
      <td>0.997516</td>
      <td>0.839506</td>
      <td>0.500000</td>
      <td>0.384192</td>
      <td>0.261351</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.182956</td>
      <td>0.023779</td>
      <td>Dredd 3D</td>
      <td>NaN</td>
      <td>0.999503</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.070486</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.066059</td>
      <td>0.125847</td>
      <td>12 Years a Slave</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.704489</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.252847</td>
      <td>0.183719</td>
      <td>2 Guns</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>0.827160</td>
      <td>0.572917</td>
      <td>0.323196</td>
      <td>0.371052</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.157175</td>
      <td>0.233625</td>
      <td>42</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>0.925926</td>
      <td>0.645833</td>
      <td>0.137984</td>
      <td>0.231656</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Your code here
```

## 2. Fit a regression model to the training data.


```python
#Your code here
```


```python
import matplotlib.pyplot as plt
%matplotlib inline
```

## 2b. Plot the training predictions against the actual data. (Y_hat_train vs Y_train)


```python
#Your code here
```

## 2c. Plot the test predictions against the actual data. (Y_hat_test vs Y_train)


```python
#Your code here
```

## 3. Calculating Bias
Write a formula to calculate the bias of a models predictions given the actual data.   
(The expected value can simply be taken as the mean or average value.)  
$Bias(\hat{f}(x)) = E[\hat{f}(x)-f(x)]$  


```python
def bias():
    pass
```

## 4. Calculating Variance
Write a formula to calculate the variance of a model's predictions (or any set of data).  
$Var(\hat{f}(x)) = E[\hat{f}(x)^2] - \big(E[\hat{f}(x)]\big)^2$


```python
def variance():
    pass
```

## 5. Us your functions to calculate the bias and variance of your model. Do this seperately for the train and test sets.


```python
#Train Set
b = None#Your code here
v = None#Your code here
#print('Bias: {} \nVariance: {}'.format(b,v))
```


```python
#Test Set
b = None#Your code here
v = None#Your code here
#print('Bias: {} \nVariance: {}'.format(b,v))
```

## 6. Describe in words what these numbers can tell you.

#Your description here (this cell is formatted using markdown)

## 7. Overfit a new model by creating additional features by raising current features to various powers.


```python
#Your Code here
```

## 8a. Plot your overfitted model's training predictions against the actual data.


```python
#Your code here
```

## 8b. Calculate the bias and variance for the train set.


```python
#Your code here
```

## 9a. Plot your overfitted model's test predictions against the actual data.


```python
#Your code here
```

## 9b. Calculate the bias and variance for the train set.


```python
#Your code here
```

## 10. Describe what you notice about the bias and variance statistics for your overfit model.

#Your description here (this cell is formatted using markdown)
