---
title:  "손실함수(Loss Function)와 최소제곱법: 회귀 모델의 최적화"
date:   2024-11-05 23:03:36 +09:00
categories: [AI, 머신러닝]
tags:
    [
        선형회귀,
        손실함수,
        최소제곱법,
        AI,
        머신러닝
    ]
use_math: true 
published: true
---

최소제곱법은 회귀 분석에서 가장 기본적인 방법으로, 실제값과 예측값 간의 차이를 최소화하는 회귀 계수를 계산하는 데 사용된다. 주어진 데이터를 가장 잘 설명하는 회귀선을 찾기 위해 잔차(Residual)의 제곱합을 최소화하는 방법이다.
<br>

손실 함수(Loss Function)
----
손실 함수(Loss Function)는 __모델이 실제값에 비해 얼마나 오차가 있는지__ 측정하는 지표로, 주어진 데이터에서 최적의 모델을 학습시키기 위해 사용된다. 다양한 손실 함수가 존재하며, 선형 회귀에서는 잔차 제곱합(RSS)이나 평균 제곱 오차(MSE)를 주로 사용한다.

<br>

잔차제곱합(Residual Sum of Squares, RSS)
----

#### 잔차 (Residual)

잔차는 실제 값과 모델이 예측한 값 간의 차이이다.



$$
e_i = y_i - \hat{y}_i
$$

- $y_i$ : 실제 값 (관측값)
- $\hat{y}_i$ : 예측 값 (모델로 계산된 값)

#### 잔차 제곱합 (RSS)

RSS는 잔차를 제곱하여 모두 더한 값이다.

$$ RSS = \sum_{i=1}^n (y_i - \hat{y}_i)^2$$

- n : 데이터의 개수
<br>

$\hat{y}_i$ 을 다음과 같이 표현할 수 있다 :


$$ RSS = \sum_{i=1}^n \left( y_i - (\beta_0 + \beta_1 x_i) \right)^2$$


- $\beta_0$ : 절편 (Intercept)
- $\beta_1$ : 기울기 또는 회귀 계수 (Slope or Coefficient)   

<br>

제곱합을 하는 이유는, 계산한 잔차가 양수나 음수가 나올 수 있는데 이때 양수와 음수를 더하면 상쇄되어 실제 오차의 크기를 제대로 측정하지 못할 수 있기 때문이다. 따라서 잔차를 제곱하여 모두 양수로 만들어준다. 

또한 이차함수는 미분이 가능하기 때문에 우리가 구한 RSS가 이차함수가 되면 RSS를 최소화하는 최소제곱법에서 미분을 쓸 수 있다. 여기서 최소제곱법은 잔차의 제곱합을 최소화하는 수학적 방법이다.

모델의 목표는 <u>관측값과 예측값 간의 차이(잔차)를 최소화하는 것이라고 할 수 있다.</u>

$$\min_{\beta_0, \beta_1} \sum_{i=1}^n \left( y_i - (\beta_0 + \beta_1 x_i) \right)^2$$

<br>

평균 제곱 오차 (Mean Squared Error, MSE)
---

MSE는 예측 값과 실제 값 사이의 차이를 제곱하여 평균을 계산하는 방식이다.

$$MSE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$

- $n$ : 데이터의 총 개수
- $y_i$ : 실제 값 (관측값)
- $\hat{y}_i$ : 모델이 예측한 값
- $(y_i - \hat{y}_i)$ : 잔차(residual), 즉 실제 값과 예측 값 간의 차이

<br>

최소제곱법
---

잔차제곱합(RSS)을 최소로 하기 위해서는 RSS가 이차함수이니 기울기가 0이 되는 지점을 찾으면 된다. 결국 __최소제곱법은 RSS를 최소화하는 $\beta_0$(절편)과  $\beta_1$(기울기)를 구하는 것__ 이다. 기울기를 구하기 위해서 미분을 하면 된다.

__(1)  RSS 를 $\beta_0$에 대해 미분__


$$RSS = \sum_{i=1}^n \left( y_i - (\beta_0 + \beta_1 x_i) \right)^2$$


$\beta_0$에 대해 미분: 

$$\frac{\partial RSS}{\partial \beta_0} = \frac{\partial}{\partial \beta_0} \sum_{i=1}^n \left( y_i - \beta_0 - \beta_1 x_i \right)^2$$


체인 룰(chain rule) 적용: 

$$\frac{\partial RSS}{\partial \beta_0} = -2 \sum_{i=1}^n \left( y_i - \beta_0 - \beta_1 x_i \right)$$


이를 0으로 설정하여 $\beta_0$을 최적화 : 

$$\sum_{i=1}^n \left( y_i - \beta_0 - \beta_1 x_i \right) = 0$$


정리하면: 

$$n \beta_0 + \beta_1 \sum_{i=1}^n x_i = \sum_{i=1}^n y_i$$

__(2)  RSS 를  $\beta_1$ 에 대해 미분__

$\beta_1$ 에 대해 미분: 

$$\frac{\partial RSS}{\partial \beta_1} = \frac{\partial}{\partial \beta_1} \sum_{i=1}^n \left( y_i - \beta_0 - \beta_1 x_i \right)^2$$


체인 룰 적용: 

$$\frac{\partial RSS}{\partial \beta_1} = -2 \sum_{i=1}^n x_i \left( y_i - \beta_0 - \beta_1 x_i \right)$$


이를 0으로 설정하여 $ \beta_1 $에 대한 최적화: 

$$\sum_{i=1}^n x_i \left( y_i - \beta_0 - \beta_1 x_i \right) = 0$$


정리하면: 

$$\beta_0 \sum_{i=1}^n x_i + \beta_1 \sum_{i=1}^n x_i^2 = \sum_{i=1}^n x_i y_i$$

__(3) 연립 방정식을 통해 $ \beta_0 $ 와  $\beta_1$  계산__

위에서 구한 두 식을 연립 방정식으로 풀어서 $\beta_0$ 와  $\beta_1$ 를 계산 :
- (식 1) $ n \beta_0 + \beta_1 \sum_{i=1}^n x_i = \sum_{i=1}^n y_i $
- (식 2) $ \beta_0 \sum_{i=1}^n x_i + \beta_1 \sum_{i=1}^n x_i^2 = \sum_{i=1}^n x_i y_i $

최종적으로 $ \beta_0 $ 와  $\beta_1$ 는 다음과 같이 계산할 수 있다.
<br>
   
기울기 : $\beta_1 = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2}$

$ \bar{x} = \frac{1}{n} \sum_{i=1}^n x_i $ : 독립 변수  x 의 평균
$ \bar{y} = \frac{1}{n} \sum_{i=1}^n y_i $ : 종속 변수  y 의 평균

절편 : $ \beta_0 = \bar{y} - \beta_1 \bar{x}$

<br>
