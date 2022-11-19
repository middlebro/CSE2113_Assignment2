# CSE2113 : Assignment 2

## Directory Structure
```bash
❯ tree .
.
├── LICENSE   # MIT license
├── README.md # README of this project
├── dataset   # dataset for this project
│   ├── default_credit_card.csv
│   └── heart.csv
├── solution.py # implementation of the template
└── template.py # assignment template

1 directory, 6 files
```

## How to run
- run solution with default_credit_card data
```python
$ python solution.py dataset/default_credit_card.csv 0.1
Number of features:  23
Number of class 0 data entries:  23364
Number of class 1 data entries:  6636

Splitting the dataset with the test size of  0.1

Decision Tree Performances
Accuracy:  0.726
Precision:  0.4065620542082739
Recall:  0.41244573082489144

Random Forest Performances
Accuracy:  0.8116666666666666
Precision:  0.6640625
Recall:  0.36903039073806077

SVM Performances
Accuracy:  0.8136666666666666
Precision:  0.6952662721893491
Recall:  0.34008683068017365
```

- run solution with heart data
```python
$ python solution.py dataset/heart.csv 0.4
Number of features:  13
Number of class 0 data entries:  499
Number of class 1 data entries:  526

Splitting the dataset with the test size of  0.4

Decision Tree Performances
Accuracy:  0.9634146341463414
Precision:  0.9629629629629629
Recall:  0.9674418604651163

Random Forest Performances
Accuracy:  0.975609756097561
Precision:  0.9812206572769953
Recall:  0.9720930232558139

SVM Performances
Accuracy:  0.9121951219512195
Precision:  0.9282296650717703
Recall:  0.9023255813953488

```

