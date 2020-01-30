<p><a href="#Vvonyye_opredeleniya">1. Задача обучения по прецедентам</a></p>
  <p><a href="#Metricheskiye_algoritmy">2. Методы восстановления регрессии</a></p>
  <p><a href="#a1NN">3. Непараметрическая регрессия</a></p>
  <p><a href="#akNN">4. Формула Надарая-Ватсона</a></p>
  <p><a href="#aLOO">5. LOWESS</a></p>
  <p><a href="#akwNN">6. Линейная регрессия</a></p>
  <p><a href="#aPW">7. Проблема мультиколлинеарности</a></p>
  <p><a href="#aPF">8. Метод главных компонентa></p>
  <p><a href="#aStolp">9. Adaboost</a></p>
  


## _Задача обучения по прецедентам_ *(основная задача ТМО)*: <a name="Vvonyye_opredeleniya"></a>

Задано множество объектов *X* и множество допустимых ответов *Y*.
Существует целевая функция ![alt text](https://latex.codecogs.com/gif.latex?y^*:&space;X\rightarrow&space;Y), значения которой 
![alt text](https://latex.codecogs.com/gif.latex?y_i&space;=&space;y^*(x_i)) известны только на конечном подмножестве объектов 
![alt text](https://latex.codecogs.com/gif.latex?\left&space;\{&space;x_1,&space;...,&space;x_l&space;\right&space;\}&space;\subset&space;X). Пары "объект-ответ" ![alt-text](https://latex.codecogs.com/gif.latex?\left&space;(&space;x_i,&space;y_i&space;\right&space;)) называются прецедентами. Совокупность пар ![alt text](https://latex.codecogs.com/gif.latex?X^l&space;=&space;\left&space;(&space;x_i,&space;y_i&space;\right&space;),&space;i&space;=&space;1,&space;...,&space;l) называется обучающей выборкой. 

Задача обучения по прецедентам заключается в том, чтобы по выборке ![alt text](https://latex.codecogs.com/gif.latex?X^l) восстановить зависимость ![alt text](https://latex.codecogs.com/gif.latex?y^*), т.е. построить решающую функцию 
![alt text](https://latex.codecogs.com/gif.latex?a:&space;X&space;\rightarrow&space;Y), которая приближала бы целевую функцию 
![alt text](https://latex.codecogs.com/gif.latex?y^*(x)), причём не только на объектах обучающей выборки, но и на всём множестве *X*.

# Методы восстановления регрессии <a name="Metricheskiye_algoritmy"></a>

*Задачей восстановления регрессии* называется задача обучения по прецедентам при ![alt text](https://latex.codecogs.com/gif.latex?Y&space;=&space;\mathbb{R}). Решающую функцию *a* называют *"функцией регрессии"*.

Пусть задана *модель регрессии* ![alt text](https://latex.codecogs.com/gif.latex?\phi(x,&space;\alpha),&space;\alpha&space;\in&space;\mathbb{R}^m), где ![alt text](https://latex.codecogs.com/gif.latex?\alpha) — вектор параметров модели. В качестве *функционала качества* используется сумма квадратов ошибок (SSE):

![alt text](https://latex.codecogs.com/gif.latex?Q%28%5Calpha%2C%20X%5El%29%20%3D%20%5Csum_%7Bi%20%3D%201%7D%5E%7Bl%7D%28%5Cphi%28x_i%2C%20%5Calpha%29%20-%20y_i%29%5E2)

**Метод наименьших квадратов** (*МНК*) находит вектор параметров ![alt text](https://latex.codecogs.com/gif.latex?\alpha^*), при котором функционал качества минимальный. Суть МНК заключается в приравнивании к нулю производной от SSE по вектору параметров ![alt text](https://latex.codecogs.com/gif.latex?\alpha):

![alt text](https://latex.codecogs.com/gif.latex?\frac{\partial&space;Q}{\partial&space;\alpha}&space;(\alpha,&space;X^l)&space;=&space;2\sum_{i&space;=&space;1}^{l}(\phi(x_i,&space;\alpha)&space;-&space;y_i)\frac{\partial&space;\phi}{\partial&space;\alpha}(x_i,&space;\alpha)&space;=&space;0)

## Непараметрическая регрессия (ядерное сглаживание) <a name="a1NN"></a>

*Непараметрическое восстановление регрессии* основано на той же идее, что и непараметрическое восстановление плотности распределения: значение ![alt text](https://latex.codecogs.com/gif.latex?a(x)) вычисляется для каждого объекта ![alt text](https://latex.codecogs.com/gif.latex?x) по нескольким ближайшим к нему объектам обучающей выборки. Близость объектов определяется согласно функции расстояния ![alt text](https://latex.codecogs.com/gif.latex?\rho(x,&space;x')), заданной на множестве объектов ![alt text](https://latex.codecogs.com/gif.latex?X).

### Формула Надарая-Ватсона <a name="akNN"></a>

Рассматривается самая простая модель регрессии ![alt text](https://latex.codecogs.com/gif.latex?\phi(x,&space;\alpha)&space;=&space;\alpha,&space;\alpha&space;\in&space;R) (*константа*). При этом, чтобы не получить тривиального решения, каждому объекту выборки ![alt text](https://latex.codecogs.com/gif.latex?x_i) назначаются *веса* согласно весовой функции ![alt text](https://latex.codecogs.com/gif.latex?w(x)). Они зависят, соответственно, от объекта ![alt text](https://latex.codecogs.com/gif.latex?x), в котором вычисляется значение ![alt text](https://latex.codecogs.com/gif.latex?a(x)&space;=&space;\phi(x,&space;\alpha)). 

Веса задаются таким образом, чтобы они убывали по мере увелечения расстояния от рассматриваемых объектов выборки до ![alt text](https://latex.codecogs.com/gif.latex?x). Для этого вводится невозрастающая, гладкая и ограниченная *функция ядра* ![alt text](https://latex.codecogs.com/gif.latex?K):

![alt text](https://latex.codecogs.com/gif.latex?w_i(x)&space;=&space;K\left&space;(&space;\frac{\rho(x,&space;x_i)}{h}&space;\right&space;)), где ![alt text](https://latex.codecogs.com/gif.latex?h) — ширина окна сглаживания. Чем меньше ![alt text](https://latex.codecogs.com/gif.latex?h), тем быстрее убывают веса ![alt text](https://latex.codecogs.com/gif.latex?w_i(x)) по мере удаления ![alt text](https://latex.codecogs.com/gif.latex?x_i) от ![alt text](https://latex.codecogs.com/gif.latex?x).

Можно сказать, что обучение регрессионной модели будет производиться отдельно в каждой точке ![alt text](https://latex.codecogs.com/gif.latex?x&space;\in&space;X). Для того, чтобы вычислить оптимальное ![alt text](https://latex.codecogs.com/gif.latex?\alpha,&space;\forall&space;x&space;\in&space;X,) необходимо воспользоваться *МНК*:

![alt text](https://latex.codecogs.com/gif.latex?Q(\alpha,&space;X^l)&space;=&space;\sum_{i&space;=&space;1}^{l}w_i(x)(\alpha&space;-&space;y_i)^2&space;\rightarrow&space;\min&space;\limits_{\alpha&space;\in&space;\mathbb{R}})

После приравнивания к нулю производной ![alt text](https://latex.codecogs.com/gif.latex?\frac{\partial&space;Q}{\partial&space;\alpha}), получается **формула ядерного сглаживания Надарая-Ватсона**:

![alt text](https://latex.codecogs.com/gif.latex?a_h(x,&space;X^l)&space;=&space;\frac{\sum\limits_{i&space;=&space;1}^{l}y_i&space;w_i(x)}{\sum\limits_{i&space;=&space;1}^{l}&space;w_i(x)})

Оптимальное ![alt text](https://latex.codecogs.com/gif.latex?h) подбирается по скользящему контролю *LOO* следующим образом:

![alt text](https://latex.codecogs.com/gif.latex?LOO(h,&space;X^l)&space;=&space;\sum_{i&space;=&space;1}^{l}(a_h(x_i,&space;\left\{&space;X^l&space;\backslash&space;x_i\right\})&space;-&space;y_i)^2&space;\rightarrow&space;\min\limits_h)

В качестве тестовой выборки был взят набор данных *Boston*, в котором нужно предсказать стоимость жилья на основе различных характеристик его расположения (загрязненность воздуха, близость к дорогам и т.д.). В реализации рассмотрена выборка по 5-му признаку — расположению жилья относительно среднего количества комнат на одно жильё (*RM*).

Ниже представлена реализация метода аппроксимации функции регрессии в одной точке выборки:

```python
def a_h(self, x):
    numerator = 0
    denominator = 0
    for i in range(self.X.shape[0]):
        numerator = numerator + self.Y[i] * self.core(self._dist(x, self.X[i]) / self.h)
        denominator = denominator + self.core(self._dist(x, self.X[i]) / self.h)
    if denominator == 0:
        alpha = 0
        return alpha
    else:
        alpha = numerator / denominator
        return alpha
```

В качестве ядер в реализации использовались квартическое и гауссовское, определённые ниже соответственно:

![alt text](https://latex.codecogs.com/gif.latex?K(x)&space;=&space;(1&space;-&space;x^2)^2[|x|&space;\leq&space;1])

![alt text](https://latex.codecogs.com/gif.latex?K(x)&space;=&space;\frac{1}{\sqrt{2\pi}}\exp(\frac{-x^2}{2}))

Сравнительная таблица для квартического и гауссовского ядер:

<table>
   <tr>
      <td align = center><b>Ядро</b></td>
      <td align = center><b>LOO_min</b></td>
      <td align = center><b>h_opt</b></td>
       <td align = center><b>SSE</b></td>
   </tr>
   
   <tr>
      <td align = center>Квартическое</td>
      <td align = center>6950.07</td>
      <td align = center>1.6900</td>
      <td align = center>17411</td>
   </tr>
    <tr>
      <td align = center>Гауссовское</td>
      <td align = center>6765.90</td>
      <td align = center>0.5</td>
      <td align = center>17317</td>
   </tr>
 </table>


### LOWESS — локально взвешенное сглаживание <a name="aLOO"></a>

Оценка Надарая-Ватсона чувствительна к большим одиночным выбросам. Для нахождения выбросов вычисляется *величина ошибки* ![alt text](https://latex.codecogs.com/gif.latex?\varepsilon_i&space;=&space;|a_h(x_i;&space;X^l&space;\setminus&space;\left&space;\{&space;x_i&space;\right&space;\})-y_i|). Чем она больше, тем в большей степени прецедент ![alt text](https://latex.codecogs.com/gif.latex?(x_i,&space;y_i)) является выбросом. Следовательно, таким прецедентам нужно понизить вес.

Отсюда возникла идея домножить веса ![alt text](https://latex.codecogs.com/gif.latex?w_i) на следующие коэффициенты:

![alt text](https://latex.codecogs.com/gif.latex?\gamma&space;_i&space;=&space;\widetilde{K}(\varepsilon&space;_i)&space;=&space;\widetilde{K}(|a_i&space;-&space;y_i|)),

где ![alt text](https://latex.codecogs.com/gif.latex?\widetilde{K}) — ещё одно ядро, вообще говоря, отличное от ![alt text](https://latex.codecogs.com/gif.latex?K).

Ниже представлен алгоритм нахождения коэффициентов ![alt text](https://latex.codecogs.com/gif.latex?\gamma_i):

1. ![alt text](https://latex.codecogs.com/gif.latex?\gamma_i&space;=&space;1,&space;i&space;=&space;1,&space;...,&space;l;)
2. **повторять пока** ![alt text](https://latex.codecogs.com/gif.latex?\gamma_i) не стабилизируются:
- вычислить *LOO* на каждом объекте: 

![alt text](https://latex.codecogs.com/gif.latex?a_i&space;=&space;a_h(x_i,&space;X^l\backslash\{x_i\})&space;=&space;\frac{\sum\limits_{j&space;=&space;1,&space;j&space;\neq&space;i}^{l}&space;y_j&space;\gamma_j&space;K(\frac{\rho(x_i,&space;x_j)}{h})}{\sum\limits_{j&space;=&space;1,&space;j&space;\neq&space;i}^{l}&space;\gamma_j&space;K(\frac{\rho&space;(x_i,&space;x_j)}{h})};)

- вычислить коэффициенты ![alt text](https://latex.codecogs.com/gif.latex?\gamma_i):

![alt text](https://latex.codecogs.com/gif.latex?\gamma&space;_i&space;=&space;\tilde{K}(|a_i&space;-&space;y_i|).)

Будем считать, что коэффициенты *стабилизированы*, когда для соседних ![alt text](https://latex.codecogs.com/gif.latex?\gamma) разность *LOO* станет меньше некоторого заданного значения.

Ядро ![alt text](https://latex.codecogs.com/gif.latex?\widetilde{K}(\varepsilon)) задаётся следующим образом:

![alt text](https://latex.codecogs.com/gif.latex?\widetilde{K}(\varepsilon)&space;=&space;K_Q\left&space;(&space;\frac{\varepsilon}{6med\left&space;(&space;\varepsilon_i&space;\right&space;)}&space;\right&space;),)

где ![alt text](https://latex.codecogs.com/gif.latex?K_Q) — квартическое ядро, ![alt text](https://latex.codecogs.com/gif.latex?med\left&space;(&space;\varepsilon_i&space;\right&space;)) — медиана вариационного ряда ошибок.

В качестве ядра в реализации использовалось квартическое, определённое ниже:

![alt text](https://latex.codecogs.com/gif.latex?K(x)&space;=&space;(1&space;-&space;x^2)^2[|x|&space;\leq&space;1])

Сравнительная таблица качества алгоритмов для квартического ядра:

<table>
   <tr>
      <td align = center><b>Алгоритм</b></td>
      <td align = center><b>LOO_min</b></td>
      <td align = center><b>h_opt</b></td>
   </tr>
   
   <tr>
      <td align = center><b>Формула Надарая-Ватсона</b></td>
      <td align = center>7355.914</td>
      <td align = center>1.690</td>
   </tr>
    <tr>
      <td align = center><b>LOWESS</b></td>
      <td align = center>969,613</td>
      <td align = center>1,55</td>
   </tr>
 </table>
 
 Пример работы алгоритма Надарая-Ватсона и LOWESS на одинаковой выборке с квартическим ядром:
![alt text](https://github.com/Mikel4u4u/TML/blob/master/image/загружено.png)
![alt text](https://github.com/Mikel4u4u/TML/blob/master/image/загружено%20(1).png)
![alt text](https://github.com/Mikel4u4u/TML/blob/master/image/загружено%20(3).png)
![alt text](https://github.com/Mikel4u4u/TML/blob/master/image/NW1.png)

### Линейная регрессия <a name="akwNN"></a>

Пусть каждый объект описывается *n* числовыми признаками ![alt text](https://latex.codecogs.com/gif.latex?f_j%28x%29%2C%20f_j%3A%20X%20%5Crightarrow%20%5Cmathbb%7BR%7D%2C%20j%20%3D%201%2C%20...%2C%20n). *Линейной моделью регрессии* называется линейная комбинация признаков с коэффициентами ![alt text](https://latex.codecogs.com/gif.latex?%5Calpha%20%5Cepsilon%20%5Cmathbb%7BR%7D%5En):

![alt text](https://latex.codecogs.com/gif.latex?%5Cphi%28x%2C%20%5Calpha%29%20%3D%20%5Csum_%7Bj%20%3D%201%7D%5E%7Bn%7D%5Calpha_jf_j%28x%29)

Введём следующие матричные обозначения:

![alt text](https://latex.codecogs.com/gif.latex?F%20%3D%20%28f_j%28x_i%29%29_%7Blxn%7D) - матрица объекты-признаки;

![alt text](https://latex.codecogs.com/gif.latex?y%20%3D%20%28y_i%29_%7Blx1%7D) - целевой вектор;

![alt text](https://latex.codecogs.com/gif.latex?%5Calpha%20%3D%20%28%5Calpha_j%29_%7Bnx1%7D) - вектор параметров.

Тогда необходимое условие минимума функционала ![alt text](https://latex.codecogs.com/gif.latex?Q%28%5Calpha%29%20%3D%20%7C%7CF%5Calpha%20-%20y%7C%7C%5E2)

![alt text](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20Q%7D%7B%5Cpartial%20%5Calpha%7D%28%5Calpha%2C%20X%5El%29%20%3D%202%5Csum_%7Bi%20%3D%201%7D%5E%7Bl%7D%28%5Cphi%28x_i%2C%20%5Calpha%29%20-%20y_i%29%5Cfrac%7B%5Cpartial%20%5Cphi%7D%7B%5Cpartial%20%5Calpha%7D%28x_i%2C%20%5Calpha%29%20%3D%200)

в матричном виде будет выглядеть следующим образом:

![alt text](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20Q%7D%7B%5Cpartial%20%5Calpha%7D%28%5Calpha%29%20%3D%202F%5ET%28F%5Calpha%20-%20y%29%20%3D%200%20%3D%3E%20F%5ETF%5Calpha%20%3D%20F%5ETy)

Полученная выше система уравнений называется ***нормальной системой*** для задачи наименьших квадратов. Решением данной системы уравнений является следующий вектор:

![alt text](https://latex.codecogs.com/gif.latex?%5Calpha%5E*%20%3D%20%28F%5ETF%29%5E%7B-1%7D%20F%5ETy)

В качестве тестовой выборки был взят набор данных *Boston*, в котором нужно предсказать стоимость жилья на основе различных характеристик его расположения (загрязненность воздуха, близость к дорогам и т.д.). В реализации рассмотрена выборка по 5-му признаку — расположению жилья относительно среднего количества комнат на одно жильё (*RM*).

Ниже представлен график функции линейной регрессии:

![alt text](https://github.com/SokolovaOksana/TML/blob/master/LinearRegression/LinearRegression.png)



### Проблема мультиколлинеарности <a name="Vvonyye_opredeleniya"></a>

Если матрица ![alt text](https://latex.codecogs.com/gif.latex?F%5ETF) имеет неполный ранг, то её обращение невозможно. В этом случае отбрасывают линейно-зависимые признаки и применяют *регуляризацию* или *метод главных компонент*.

Часто встречается на практике, что матрица ![alt text](https://latex.codecogs.com/gif.latex?F%5ETF) близка к вырожденной. Признаком мультиколлинеарности является наличие у матрицы ![alt text](https://latex.codecogs.com/gif.latex?F%5ETF) собственных значений, близких к нулю.

***Числом обусловленности*** матрицы ![alt text](https://latex.codecogs.com/gif.latex?F%5ETF) называется следующая величина:

![alt text](https://latex.codecogs.com/gif.latex?%5Cmu%28F%5ETF%29%20%3D%20%5Cleft%20%5C%7C%20F%5ETF%20%5Cright%20%5C%7C%5Cleft%20%5C%7C%20%28F%5ETF%29%5E%7B-1%7D%20%5Cright%20%5C%7C%20%3D%20%5Cfrac%7B%5Clambda_%7Bmax%7D%7D%7B%5Clambda_%7Bmin%7D%7D)

где ![alt text](https://latex.codecogs.com/gif.latex?%5Clambda_%7Bmax%7D%2C%20%5Clambda_%7Bmin%7D) — максимальное и минимальное собственные значения матрицы ![alt text](https://latex.codecogs.com/gif.latex?F%5ETF).

Матрица ![alt text](https://latex.codecogs.com/gif.latex?F%5ETF) считается плохо обусловленной (близкой к вырожденной), если ![alt text](https://latex.codecogs.com/gif.latex?%5Cmu%28F%5ETF%29%5Cgeqslant%2010%5E2%20..%2010%5E4).
### Метод главных компонент <a name="aPW"></a>
Метод Главных Компонент (англ. Principal Components Analysis, PCA) — один из основных способов уменьшить размерность данных, потеряв наименьшее количество информации. Изобретен К. Пирсоном (англ.Karl Pearson) в 1901 г. Применяется во многих областях, таких как распознавание образов, компьютерное зрение, сжатие данных и т. п. Вычисление главных компонент сводится к вычислению собственных векторов и собственных значений ковариационной матрицы исходных данных или к сингулярному разложению матрицы данных. 
[ссылке](PCA.ipynb) , [ссылке2](PCA2.ipynb)

### Решающие деревья <a name="aPF"></a>

Решающие деревья воспроизводят логические схемы, позволяющие получить окончательное решение о классификации объекта с
помощью ответов на иерархически организованную систему вопросов. Каждой из вершин дерева за исключением листьев соответствует
некоторый вопрос, подразумевающий несколько вариантов ответов,соответствующих выходящим рёбрам. В зависимости от выбранного варианта ответа осуществляется переход к вершине следующего уровня. Концевым вершинам поставлены в соответствие метки,указывающие на отнесение распознаваемого объекта к одному из классов.
Распознавание с помощью решающих деревьев. 
Предположим,что бинарное дерево T используется для распознавания объектов, описываемых набором признаков <a href="https://www.codecogs.com/eqnedit.php?latex=X_{1},&space;.&space;.&space;.&space;,&space;X_{n}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X_{1},&space;.&space;.&space;.&space;,&space;X_{n}" title="X_{1}, . . . , X_{n}" /></a>.
Каждой вершине ν дерева T ставится в соответствие предикат, касающийся значения одного из признаков. Непрерывному признаку
<a href="https://www.codecogs.com/eqnedit.php?latex=X_{j}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X_{j}" title="X_{j}" /></a> соответствует предикат вида <a href="https://www.codecogs.com/eqnedit.php?latex=X_{j}&space;\geq&space;\partial_{j}^{v}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X_{j}&space;\geq&space;\partial_{j}^{v}" title="X_{j} \geq \partial_{j}^{v}" /></a> , где <a href="https://www.codecogs.com/eqnedit.php?latex=\partial_{j}^{v}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\partial_{j}^{v}" title="\partial_{j}^{v}" /></a> - некоторый пороговый параметр.
Пример работы на выборке "Ирисы Фишера" :

![alt text](https://github.com/Mikel4u4u/TML/blob/master/image/tree.png)

![alt text](https://github.com/Mikel4u4u/TML/blob/master/image/iris_tree.png)

## Композиция алгоритмов
Определение
Композицией алгоритмов ![alt text](http://latex.codecogs.com/gif.latex?a_{t}(x)&space;=&space;C(b_{t}(x)),&space;t&space;=&space;1,&space;...&space;,T)  , называется суперпозиция базовых алгоритмов (алгоритмических
операторов) <img src="http://latex.codecogs.com/gif.latex?b_{t}&space;:&space;X&space;\rightarrow&space;R" title="b_{t} : X \rightarrow R" /> , корректирующей операции <img src="http://latex.codecogs.com/gif.latex?F&space;:&space;R^{T}&space;\rightarrow&space;R" title="F : R^{T} \rightarrow R" /> и решающего правила <img src="http://latex.codecogs.com/gif.latex?F&space;:&space;R^{T}&space;\rightarrow&space;R" title="C : R \rightarrow Y" /> :

<img src="http://latex.codecogs.com/gif.latex?a(x)&space;=&space;C(F(b_{1}(x),...&space;,&space;b_{T}&space;(x));&space;x&space;\epsilon&space;X;" title="a(x) = C(F(b_{1}(x),... , b_{T} (x)); x \epsilon X;" />

где R  пространство оценок.
### Adaboost <a name="aStolp"></a>



