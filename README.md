
# Simple Knowledge Distillation in PyTorch

A simple and educational implementation of **Knowledge Distillation (KD)** using PyTorch on the CIFAR-10 dataset.

In this project, a larger CNN (Teacher) transfers its knowledge to a smaller CNN (Student). The Student is first trained conventionally to establish a baseline and is then retrained using Knowledge Distillation to investigate whether learning from a Teacher can improve performance.

---

## Overview

Knowledge Distillation is a model compression technique introduced by

Distilling the Knowledge in a Neural Network

in which a smaller network (Student) learns not only from the ground-truth labels but also from the soft predictions of a larger network (Teacher).

This repository demonstrates:

* Building Teacher and Student CNN architectures
* Conventional supervised training
* Knowledge Distillation training
* Comparison between baseline Student and KD Student
* Analysis of Cross-Entropy and KD losses

The project is intended for **educational purposes** and to provide an intuitive introduction to Knowledge Distillation.

---

# Project Workflow

The project follows the pipeline below:

```text
                  Stage 1

        CIFAR-10 Images + Labels
                    │
                    ▼

            Teacher CNN
          (larger network)

                    │
           Conventional Training
                    │
                    ▼

          Save Best Teacher
               teacher.pt


─────────────────────────────────────


                  Stage 2

        CIFAR-10 Images + Labels
                    │
                    ▼

            Student CNN
         (smaller network)

                    │
           Conventional Training
                    │
                    ▼

          Baseline Student


─────────────────────────────────────


                  Stage 3

                 Teacher
                (Frozen)
                    │
                    ▼

Images ───► Student CNN

Student learns from:

1. Ground Truth Labels
2. Teacher Soft Predictions

using:

CrossEntropy Loss
+
Knowledge Distillation Loss
```

---

# Dataset

The experiments are performed on the **CIFAR-10** dataset.

CIFAR-10 contains:

* 60,000 RGB images
* Image size: 32 × 32
* 10 classes:

```text
airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck
```

---

# Teacher Network

The Teacher is a relatively larger CNN inspired by the VGG architecture.

Characteristics:

* Multiple convolutional blocks
* Deeper architecture
* Larger number of trainable parameters
* Higher representational capacity

The Teacher is first trained conventionally using CrossEntropyLoss.

The best Teacher model (based on validation loss) is saved as:

```text
teacher.pt
```

In this project:

* Training epochs: 15
* Final validation accuracy: approximately 83%
* The Teacher is intentionally kept relatively small and underfitted due to educational and computational constraints.

---

# Student Network

The Student is a simplified version of the Teacher.

Characteristics:

* Fewer convolutional layers
* Fewer parameters
* Lower computational cost
* Faster training

The Student is trained in two different ways:

1. Conventional supervised training
2. Knowledge Distillation training

This allows direct comparison between:

* Student trained only from labels
* Student trained from labels + Teacher knowledge

---

# Conventional Training

The Teacher and the baseline Student are trained using:

```python
loss = CrossEntropyLoss(outputs, labels)
```

The training procedure is:

```text
Images
   │

Student / Teacher

   │

Predictions

   │

CrossEntropy Loss

   │

Ground Truth Labels
```

---

# CrossEntropy Loss

CrossEntropyLoss is the standard loss function for multiclass classification.

For a classification problem with (C) classes:

[
L_{CE}
======

-\sum_{i=1}^{C}
y_i
\log(p_i)
]

where:

* (y_i) is the one-hot encoded label
* (p_i) is the predicted probability

The loss encourages the network to assign a high probability to the correct class.

---

### Why CrossEntropy?

The CIFAR-10 task is:

* Single-label classification
* Mutually exclusive classes
* One image belongs to exactly one category

Therefore, CrossEntropyLoss is the natural and most widely used choice.

---

# Knowledge Distillation

After the Teacher is trained:

```python
teacher = torch.load('teacher.pt')
teacher.eval()
```

the Teacher is frozen and used to supervise a newly initialized Student.

The Student learns from:

1. Ground-truth labels
2. Teacher soft predictions

The Teacher parameters are not updated:

```python
with torch.no_grad():
    teacher_outputs = teacher(images)
```

Only the Student receives gradients.

---

# Knowledge Distillation Loss

The KD loss combines:

* CrossEntropy Loss
* KL Divergence

The overall loss is:

[
L_{KD}
======

\alpha T^2
D_{KL}(P_t||P_s)
+
(1-\alpha)
L_{CE}
]

where:

* (P_t): Teacher probability distribution
* (P_s): Student probability distribution
* (T): Temperature
* (\alpha): weighting factor

In this project:

```python
T = 10
alpha = 0.6
```

Thus:

* 60% of the supervision comes from the Teacher
* 40% comes from the ground-truth labels

---

# Dark Knowledge

One of the main ideas behind Knowledge Distillation is **Dark Knowledge**.

Instead of learning only:

```text
Dog = 1
Everything else = 0
```

the Student learns:

```text
Dog   : 0.60
Wolf  : 0.25
Fox   : 0.10
Cat   : 0.05
```

The Teacher provides information about:

* inter-class similarities,
* uncertainty,
* semantic relationships.

This richer supervision often leads to:

* improved generalization,
* smoother optimization,
* higher accuracy,
* reduced overfitting.

---

# Conventional Training vs Knowledge Distillation

## Conventional Training

```text
Images

↓

Student

↓

Predictions

↓

CrossEntropy Loss

↓

Ground Truth Labels
```

The Student learns only from:

```text
labels
```

---

## Knowledge Distillation Training

```text
                      Teacher (Frozen)
                    ↗

Images → Student → Predictions

                    ↘

                   KD Loss

KD Loss =
    CrossEntropy(Student, Labels)
  + KL(Student, Teacher)
```

The Student learns from:

```text
labels
+
teacher predictions
```

---

# Experimental Results

Student alone => 83% within 15 epochs / 
Student while learning from the teacher => 85% within 15 epcohs / 
A  noticable increase

---
### ReadMe is mostly generated by GPT! Check the important info!


