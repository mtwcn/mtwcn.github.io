<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

Activation

Relu 
----

#### 公式：  

$$ y = \max_{}(0,x) $$

#### 函数：

<p align="left">
![](/library/img/ai_ml/operator_relu.png) </p>

#### 实现

Relu6
----

#### 公式：  

$$ y = \ min (6, \max_{}(0,x)) $$ 

#### 函数：

<p align="left">
![](/library/img/ai_ml/operator_relu6.png) </p>

#### 实现

Relu1
----

#### 公式：  

$$ y = \min(1, \max_{}(-1,x))  $$

#### 函数：

<p align="left">
![](/library/img/ai_ml/operator_relu1.png) </p>

#### 实现

 	

TANH
----

#### 公式：  

$$ sinh(x) = {e^x - e^{-x}  \over 2}  \ \ \ \ cosh(x) = {e^x - e^{-x}  \over 2}  $$
$$ tanh(x) = {sinh(x) \over cosh(x)} = {e^x - e^{-x}  \over e^x + e^{-x} } $$

#### 函数：

<p align="left">
![](/library/img/ai_ml/operator_tanh.png) </p>

#### 实现

Softmax
----

#### 公式：  

$$ \sigma (\mathbf {z} )_{j}={\frac {e^{z_{j}}}{\sum _{k=0}^{K}e^{z_{k}}}} $$


#### 实现

Logistic
----

#### 公式：  

$$ y = {1 \over 1 + e^{-x}}   $$

#### 函数：

<p align="left">
![](/library/img/ai_ml/operator_logistic.png) </p>

#### 实现


