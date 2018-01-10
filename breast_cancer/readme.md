

```python
import pandas as pd
import numpy as np

column_names = ['Sample', 'Clump Thinkness', 'Uniformity of Cell Size', 
               'Uniformity of Cell Shape', 'Marginal Adhesion', 
               'Single Epithelial Cell Size', 'Bare Nuclei', 
               'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
data = pd.read_csv('breast-cancer-wisconsin.data', names=column_names)

data = data.replace(to_replace='?', value=np.nan) # 将？替换成缺省值
data = data.dropna(how='any') # 丢弃带有缺省值的数据
data.shape
```




    (683, 11)




```python
pd.DataFrame(data.head(10))
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sample</th>
      <th>Clump Thinkness</th>
      <th>Uniformity of Cell Size</th>
      <th>Uniformity of Cell Shape</th>
      <th>Marginal Adhesion</th>
      <th>Single Epithelial Cell Size</th>
      <th>Bare Nuclei</th>
      <th>Bland Chromatin</th>
      <th>Normal Nucleoli</th>
      <th>Mitoses</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000025</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1002945</td>
      <td>5</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>7</td>
      <td>10</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1015425</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1016277</td>
      <td>6</td>
      <td>8</td>
      <td>8</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>7</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1017023</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1017122</td>
      <td>8</td>
      <td>10</td>
      <td>10</td>
      <td>8</td>
      <td>7</td>
      <td>10</td>
      <td>9</td>
      <td>7</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1018099</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>10</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1018561</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1033078</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1033078</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.model_selection import train_test_split

# 随机采样 25% 用于测试，剩下 75% 用于训练
X_train, X_test, y_train, y_test = train_test_split(data[column_names[1:10]],
                                                    data[column_names[10]],
                                                    test_size=0.25,
                                                    random_state=33)

y_train.value_counts()
```




    2    344
    4    168
    Name: Class, dtype: int64




```python
y_test.value_counts()
```




    2    100
    4     71
    Name: Class, dtype: int64




```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

lr = LogisticRegression()
sgdc = SGDClassifier(max_iter=15)
lr.fit(X_train, y_train)
lr_y_predict = lr.predict(X_test)
sgdc.fit(X_train, y_train)
sgdc_y_predict = sgdc.predict(X_test)
```


```python

print('Accuracy of LR Classifier:', lr.score(X_test, y_test))
print(classification_report(y_test, lr_y_predict, target_names=['Benign', 'Malignant']))
```

    Accuracy of LR Classifier: 0.988304093567
                 precision    recall  f1-score   support
    
         Benign       0.99      0.99      0.99       100
      Malignant       0.99      0.99      0.99        71
    
    avg / total       0.99      0.99      0.99       171
    
    


```python
print('Accuracy of SGD Classifier:', sgdc.score(X_test, y_test))
print(classification_report(y_test, sgdc_y_predict, target_names=['Benign', 'Malignant']))
```

    Accuracy of SGD Classifier: 0.982456140351
                 precision    recall  f1-score   support
    
         Benign       1.00      0.97      0.98       100
      Malignant       0.96      1.00      0.98        71
    
    avg / total       0.98      0.98      0.98       171
    
    
