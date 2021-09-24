# L-Layer-Neural-Network in C++
L-Layer Neural Network implemented in C++ for self challange with "heart.csv" dataset. The dataset is manipulated a little bit to turn the problem into a binary classification problem. It does not produce very meaningful results but effects of hyperparameter tuning can be observed easily. 

## Used Libraries
<ul>
<li>Eigen</li>
</ul> 

## Some Results 
4 layers are used. 8-4-2-1 neurons are used from left to right.
Models:
<ol>
<li>Learning rate: 0.05, Iterations: 100</li>
<li>Learning rate: 20, Iterations: 15</li>
<li>Learning rate: 0.001, Iterations: 100</li>
</ol>

|        | Correct | False |
|:------:|:-------:|:-----:|
| Model1 |    22   |   25  |
| Model2 |    8    |   39  |
| Model3 |    10   |   37  |
