
# Решающее дерево


```python
from sklearn.datasets import make_blobs, make_moons
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
%matplotlib inline

x, y = make_moons(n_samples=300, noise=0.1)
# x, y = make_blobs(n_samples=150, n_features=2, cluster_std=1.2, centers=2)

plt.scatter(x[:,0], x[:,1], c = y)
```




    <matplotlib.collections.PathCollection at 0x1a17e7cbe0>




![png](output_1_1.png)


Отметим, что решающее дерево состоит из вершин, в которых записывает некоторое условие, а в листах дерева - прогноз. Условия во внутренних вершинах выбираются простыми. Наиболее частый вариант - проверка лежит ли значение некоторого признака $x^j$ левее порога $t$:

$[x^j \leq t]$

Легко заметить, что такое условие зависит только от одного признака. Прогноз в листе является вещественным числом, если решается задача регрессии. Если же решается задача классификации, то в качестве прогноза выступает или класс или распределение вероятностей классов.

Запишем некоторую функцию ошибки следующим образом: есть набор данных $X_m$, есть номер рассматриваемого признака $j$ и есть порог $t$.

$L(X_m, j, t) \rightarrow \underset{j,t}{min}$

Осуществим перебор всех возможных признаков $j$ и порогов $t$ для этих признаков. Каждый раз исходное множество будет разбиваться на два подмножества:

$X_l = \{x \in X_m | [ x^j \leq t ] \}$ и $X_к = \{x \in X_m | [ x^j > t ] \}$

Такую процедуру можно продолжить для каждого получившегося множества (рекурсивно вызывать функцию деления для получающихся подмножеств).

Необходимо отметить, что если вершина была объявлена листом, необходимо сделать прогноз для этого листа. Для задачи регрессии берется среднее значение по этой выборке,
для задачи классификации возвращается тот класс, которые наиболее популярен в выборке. Можно указывать вероятность классов в листе.

Для каждого разбиения необходимо расчитывать функцию потерь:

$L(X_m, j, t) = \frac{|X_l|}{|X_m|}H(X_l) + \frac{|X_r|}{|X_m|}H(X_r)$,

где $H(X)=\sum\limits_{k=1}^{K} p_k(1 - p_k)$ - критерий информативности Джинни.

$p_k$ - доля объектов класса $k$ в выборке X:

$p_k=\frac{1}{|X|}\sum\limits_{i \in X}{[y_i = k]}$

В задаче работаем только с числовыми данными и строим дерево классификации.

Данные сгенерированы случайным образом.


## Задание 1

Напишите функцию, которая получает на вход матрицу данных ($x$) и целевые переменные ($y$), на выходе функция возвращает номер признака ($i$), порог разбиения($t$), значение в листовой вершине слева ($y_l$) и значение в листовой вершине справа ($y_r$).
Постарайтесь учесть, что пороги разбиения должны лежать строго по середине между ближайшими обектами.

Необходимо учесть:

1. Разбиений не требуется, если в получившемся множестве находятся объекты одного класса.
2. Количество различных классов объектов в целевой переменной может быть больше двух.



```python
def getGini(left, right, edge):
    allValues = left + right
    allValuesCount = float(len(allValues))
    allCategories = list(set([value[2] for value in allValues]))
    
    gini = 0.0
    for group in (left, right):
        groupLen = float(len(group))
        score = 0.0
        for category in allCategories:
            catRelation = [key[2] for key in group].count(category) / groupLen
            score += catRelation * catRelation

        gini += (1 - score) * (groupLen / allValuesCount)
    return gini
```


```python
def calculateLoss(dimensionTuples, edge):
    left = []
    right = []
    
    for value in dimensionTuples:
        targetList = left if value[1] <= edge else right
        targetList.append(value)

    return getGini(left, right, edge)
```


```python
def getDimensionEdge(positionsList, pointValues):
    # create tuples for dimension like (index, position, value)
    dimensionTuples = []
    for key, position in enumerate(positionsList):
        dimensionTuples.append((key, position, pointValues[key]))
    dimensionTuples.sort(key=lambda v: v[1])
    
    minGini, index = 1, -1
    for key in range(len(dimensionTuples) - 1):
        left = dimensionTuples[key]
        right = dimensionTuples[key + 1]
        edge = (left[1] + right[1]) / 2
        
        gini = calculateLoss(dimensionTuples, edge)
        if (gini < minGini):
            minGini, index = gini, key

    return minGini, dimensionTuples[index], dimensionTuples[index + 1];
```


```python
def getLeaf(y):
    categoriesList = list(set(y));
    return -1 if len(categoriesList) > 1 else int(categoriesList[0])
```


```python
def splitByEdge(x, y, dimension, edge):
    xTargets = {
        'x': [],
        'y': []
    }
    yTargets = {
        'x': [],
        'y': []
    }
    for key, values in enumerate(x):
        targets = xTargets if values[dimension] <= edge else yTargets
        targets['x'].append(values)
        targets['y'].append(y[key])

    return [xTargets, yTargets]
```


```python
def getEdge(valuesSet):
    if(type(valuesSet) is int):
        return valuesSet
    
    x = valuesSet['x']
    y = valuesSet['y']
    
    leaf = getLeaf(y);
    
    if (not leaf == -1):
        return leaf;
    
    minDimensionKey, minGini, minLeft, minRight = -1, 1, 0, 0
    for dimensionKey in range(len(x[0])):
        dimensionValues = [values[dimensionKey] for values in x]
        gini, left, right = getDimensionEdge(dimensionValues, y)
        if (gini < minGini):
            minDimensionKey, minGini, minLeft, minRight = dimensionKey, gini, left, right
    
    edge = (minLeft[1] + minRight[1]) / 2
    
    left, right = splitByEdge(x, y, minDimensionKey, edge);
    leftLeaf = getLeaf(left['y'])
    rightLeaf = getLeaf(right['y'])
    
    return {
        'dimension': minDimensionKey,
        'edge': edge,
        'leftValue': minLeft[1],
        'rightValue': minRight[1],
        'gini': minGini,
        'left': left,
        'right': right,
    };

firstSplit = getEdge({
    'x': x,
    'y': y
})
```

После первого разбиения мы считаем, что в листе дерева будет находиться класс объектов, которых больше всего в этом множестве. Реализуйте классификатор дерева решений в виде функции, которая получает на вход признаки объекта - $x$, на выходе метка класса - $y_{pred}$. Величину порога деления возьмите с точностью до 5 знака после запятой.

Визуализируйте получившиеся результаты классификатора на плоскости. Для этого воспользуйтесь кодом ниже, чтобы построить поверхность. $t$ - порог разбиения $i$ - номер признака.


```python
def tree_clf(x):
    y_pred = list()  
    for it in x.transpose():
        y_pred.append(1 if it[firstSplit['dimension']] <= firstSplit['edge'] else 0)
    return np.array(y_pred)

h = .02
x0_min, x0_max = np.min(x[:,0]) - 1, np.max(x[:,0]) + 1
x1_min, x1_max = np.min(x[:,1]) - 1, np.max(x[:,1]) + 1
xx0, xx1 = np.meshgrid(np.arange(x0_min, x0_max, h),
                         np.arange(x1_min, x1_max, h))

Z = tree_clf(np.stack((xx0.ravel(),xx1.ravel())))

Z = Z.reshape(xx0.shape)
cm = plt.cm.RdBu
plt.contourf(xx0, xx1, Z, cmap=cm, alpha=.8)
plt.scatter(x[:,0], x[:,1], c = y)

```




    <matplotlib.collections.PathCollection at 0x1a182e3748>




![png](output_11_1.png)


Проведите процесс разбиение в получившихся листах дерева до глубины 2. Перепишите функцию классификатора. Визуализируйте результаты классификации.


```python
# decisionsTree:
# int or {
#     dimension: 1,
#     edge: 1.5,
#     left: 2, # particular leaf (int)
#     right: {...decisionsTree} # decisionsTree (dict)
# }
def categorizePoint(point, decisionTree):
    dimension = decisionTree['dimension']
    edge = decisionTree['edge']
    left = decisionTree['left']
    right = decisionTree['right']
    target = left if point[dimension] <= edge else right
    
    if (type(target) is dict):
        return categorizePoint(point, target);
    else:
        return target;
```


```python
def categorizePoints(x, decisionTree):
    y_pred = list()  
    for it in x.transpose():
        y_pred.append(categorizePoint(it, decisionTree))
    return np.array(y_pred)
```


```python
def printDecisionsTree(x, decisionTree):
    h = .02
    x0_min, x0_max = np.min(x[:,0]) - 1, np.max(x[:,0]) + 1
    x1_min, x1_max = np.min(x[:,1]) - 1, np.max(x[:,1]) + 1
    xx0, xx1 = np.meshgrid(np.arange(x0_min, x0_max, h),
                             np.arange(x1_min, x1_max, h))

    Z = categorizePoints(np.stack((xx0.ravel(),xx1.ravel())), decisionTree)

    Z = Z.reshape(xx0.shape)
    cm = plt.cm.RdBu
    plt.contourf(xx0, xx1, Z, cmap=cm, alpha=.8)
    plt.scatter(x[:,0], x[:,1], c = y)
```


```python
if (type(firstSplit['left']) is not int):
    firstSplit['left'] = getEdge(firstSplit['left'])
    firstSplit['left']['left'] = 30
    firstSplit['left']['right'] = 40

if (type(firstSplit['right']) is not int):
    firstSplit['right'] = getEdge(firstSplit['right'])
    firstSplit['right']['left'] = 50
    firstSplit['right']['right'] = 60

printDecisionsTree(x, firstSplit)
```


![png](output_16_0.png)



```python
def getDesisionTree(valuesSet):
    decision = getEdge(valuesSet);
    
    if (type(decision) is int):
        return decision;
    else:
        return {
            'dimension': decision['dimension'],
            'edge': decision['edge'],
            'left': getDesisionTree(decision['left']),
            'right': getDesisionTree(decision['right']),
        }

tree = getDesisionTree({
    'x': x,
    'y': y,
})

import pprint
pprint.pprint(tree)

printDecisionsTree(x, tree)
```

    {'dimension': 1,
     'edge': 0.3021680937858746,
     'left': {'dimension': 0,
              'edge': -0.4718758656014028,
              'left': 0,
              'right': {'dimension': 1,
                        'edge': -0.05354186660046033,
                        'left': 1,
                        'right': {'dimension': 0,
                                  'edge': 0.5161290259456924,
                                  'left': 1,
                                  'right': {'dimension': 0,
                                            'edge': 1.3309138561583729,
                                            'left': 0,
                                            'right': 1}}}},
     'right': {'dimension': 0,
               'edge': 1.4915406522515053,
               'left': {'dimension': 1,
                        'edge': 0.5086553012860503,
                        'left': {'dimension': 0,
                                 'edge': 0.42188453997544684,
                                 'left': {'dimension': 0,
                                          'edge': -0.3307353541916291,
                                          'left': 0,
                                          'right': 1},
                                 'right': 0},
                        'right': 0},
               'right': 1}}



![png](output_17_1.png)


## Задание №2. Кросс-проверка и  подбор параметров для дерева решений

В задании воспользуемся готовой функцией для построения дерева решений DecisionTreeClassifier. Использовать будем данные с платформы kaggle, где проводится классификация пациентов на здоровых ('Normal') и с заболеванием ('Abnormal'). Подробности о признаках тут: https://www.kaggle.com/uciml/biomechanical-features-of-orthopedic-patients/home
Необходимо построить кросс-проверку (cross_val_score в sklearn.cross_validation ) алгоритма "Дерево решений" на 5 подвыборках, причем дерево должно не ограничиваться по глубине (max_depth), в листовой вершине дерева должен быть хотя бы один объект (min_samples_leaf), минимальное количество объектов должно быть не менее 2 для разбиения (min_samples_split) и random_state=42.
Оцените результаты получившиеся на валидационной выборке с использованием F1-меры, точности и полноты (в параметре scoring к методу cross_val_score необходимо использовать 'f1_weighted', 'precision_weighted', 'recall_weighted'). 

Точность и полнота позволяют провести оценку классификации.
Точность (Precision) - это отношение количества правильно классифицированных объектов класса к количеству объектов, которые классифицированы как данный класс.
Полнота (Recall) - это отношение количества правильно классифицированных объектов класса к количеству объектов данного класса в тестовой выборке.
Введем следующие понятия:

$TP$  — истино-положительное решение;
$TN$ — истино-отрицательное решение;
$FP$ — ложно-положительное решение;
$FN$ — ложно-отрицательное решение.

Тогда 

$Precision = \frac{TP}{TP+FP}$, 

$Recall = \frac{TP}{TP+TN}$.

Точность и полнота изменяются в интервале [0, 1]. Чем выше точность и полнота, тем лучше классификатор. Однако, по двум метрикам очень сложно чудить о качестве классификации, поэтому используют F1-меру:

$F1 = \frac{2 \times Precision \times Recall}{Precision + Recall}$

В нашем задании будем использовать взвешенную сумму F1-меры, точности и полноты для каждого класса.


```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics.classification import f1_score, precision_score, recall_score
```


```python
df = pd.read_csv('column_2C_weka.csv')
```


```python
clf = DecisionTreeClassifier(min_samples_leaf=1, min_samples_split=2, random_state=42)

classes = df.values[:,6]
values = df.values[:,:6]

for scoring in ['f1_weighted', 'precision_weighted', 'recall_weighted']: 
    print(scoring + ': ', cross_val_score(clf, x, y, scoring=scoring, cv=5))
```

    f1_weighted:  [1.         0.94998611 1.         1.         1.        ]
    precision_weighted:  [1.         0.95050056 1.         1.         1.        ]
    recall_weighted:  [1.   0.95 1.   1.   1.  ]


Следует отметить, что по одной метрике не очень удобно проводить оценку. В этой связи, лучшим вариантом кросс-проверки будет использование функции cross_validate из sklearn.model_selection. В нее можно передать несколько метрик и оценивать каждый шаг кросс проверки по нескольким метрикам, причем для тренировочного (return_train_score=True) и валидационного набора данных.
Оцените результаты на тренировочном и тестовом наборе. Объясните получившиеся результаты.



```python
from sklearn.model_selection import cross_validate
```


```python
score = cross_validate(clf, x, y, scoring=scoring, cv=5, return_train_score=True)

print('Время подсчета качества оченки результатов выборки для тестового набора данных', score['fit_time'])
print('Время подсчета качества оченки результатов выборки для тренировочного набора данных', score['score_time'])
print('Оценка качества обработки выборки на тестовых данных', score['test_score'])
print('Оценка качества обработки выборки на тренировочных данных', score['train_score'])
```

    Время подсчета качества оченки результатов выборки для тестового набора данных [0.00145912 0.00049424 0.00047398 0.00081611 0.00051999]
    Время подсчета качества оченки результатов выборки для тренировочного набора данных [0.00078511 0.00059295 0.00076985 0.00089693 0.00055289]
    Оценка качества обработки выборки на тестовых данных [1.   0.95 1.   1.   1.  ]
    Оценка качества обработки выборки на тренировочных данных [1. 1. 1. 1. 1.]


Подбор параметров для модели можно проводить с помощью GridSearchCV из sklearn.model_selection. Возьмите параметр максимальной глубины дерева от 1 до 20. Относительно этого параметра вычислите средевзвешенную F1-меру для 5 разбиений. Постройте график зависимости F1-меры от максимальной глубины дерева. Получить результаты построения можно методом cv_results_.

Проведите оценку результатов. Почему на тестовой выборке F1-мера не увеличивается. 


```python
from sklearn.model_selection import GridSearchCV
```


```python
dTC = DecisionTreeClassifier(min_samples_leaf=1, min_samples_split=2, random_state=42, max_depth=1)

tree_para = {'max_depth':[depth for depth in range(1, 21)]}
clf = GridSearchCV(dTC, tree_para, cv=5, scoring='f1_weighted')
clf.fit(values, classes)

plt.scatter([param['max_depth'] for param in clf.cv_results_['params']], clf.cv_results_['mean_test_score'])
```




    <matplotlib.collections.PathCollection at 0x1a19f8f278>




![png](output_28_1.png)


То же самое проведите для параметра min_samples_split от 2 до 20.

Проведите оценку результатов. 


```python
dTC = DecisionTreeClassifier(min_samples_leaf=1, random_state=42, max_depth=1)

tree_para = {'min_samples_split':[depth for depth in range(2, 21)]}
clf = GridSearchCV(dTC, tree_para, cv=5, scoring='f1_weighted')
clf.fit(values, classes)

plt.scatter([param['min_samples_split'] for param in clf.cv_results_['params']], clf.cv_results_['mean_test_score'])
```




    <matplotlib.collections.PathCollection at 0x1a199817b8>




![png](output_30_1.png)


То же самое проведите для параметра min_samples_leaf от 1 до 10.

Проведите оценку результатов. 


```python
dTC = DecisionTreeClassifier(min_samples_split=2, random_state=42, max_depth=1)

tree_para = {'min_samples_leaf':[depth for depth in range(1, 11)]}
clf = GridSearchCV(dTC, tree_para, cv=5, scoring='f1_weighted')
clf.fit(values, classes)

plt.scatter([param['min_samples_leaf'] for param in clf.cv_results_['params']], clf.cv_results_['mean_test_score'])
```




    <matplotlib.collections.PathCollection at 0x1a1a392f60>




![png](output_32_1.png)


Выяснили, что качество разбиения коррелирует только с max_depth, при этом глубже 10 делать смысла нет потому,
что это максимальная глубина дерева для этой выборки.

Качество кросс-валидации не меняется в зависимости от минимального количества листов и разбиений.
