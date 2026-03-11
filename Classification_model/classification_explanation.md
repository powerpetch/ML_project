# Classification Model - คำอธิบายทุกโมเดลในโปรเจกต์

## สารบัญ

1. [ภาพรวมโปรเจกต์](#1-ภาพรวมโปรเจกต์)
2. [ชุดข้อมูลและการเตรียมข้อมูล](#2-ชุดข้อมูลและการเตรียมข้อมูล)
3. [Metrics ที่ใช้วัดผล](#3-metrics-ที่ใช้วัดผล)
4. [โมเดลที่ 1: Softmax Regression](#4-โมเดลที่-1-softmax-regression)
5. [โมเดลที่ 2: Logistic Regression (Sklearn)](#5-โมเดลที่-2-logistic-regression-sklearn)
6. [โมเดลที่ 3: Decision Tree](#6-โมเดลที่-3-decision-tree)
7. [โมเดลที่ 4: Random Forest](#7-โมเดลที่-4-random-forest)
8. [โมเดลที่ 5: SVM (Support Vector Machine)](#8-โมเดลที่-5-svm-support-vector-machine)
9. [โมเดลที่ 6: Random Forest + PCA](#9-โมเดลที่-6-random-forest--pca)
10. [โมเดลที่ 7: SVM + PCA](#10-โมเดลที่-7-svm--pca)
11. [โมเดลที่ 8: K-Means Clustering](#11-โมเดลที่-8-k-means-clustering)
12. [โมเดลที่ 9: Agglomerative Clustering](#12-โมเดลที่-9-agglomerative-clustering)
13. [โมเดลที่ 10: Perceptron / SLP](#13-โมเดลที่-10-perceptron--slp)
14. [โมเดลที่ 11: MLP (Multi-Layer Perceptron)](#14-โมเดลที่-11-mlp-multi-layer-perceptron)
15. [โมเดลที่ 12: Gradient Boosting](#15-โมเดลที่-12-gradient-boosting)
16. [โมเดลขั้นสูง: XGBoost](#16-โมเดลขั้นสูง-xgboost)
17. [โมเดลขั้นสูง: LightGBM](#17-โมเดลขั้นสูง-lightgbm)
18. [โมเดลขั้นสูง: Stacking Ensemble](#18-โมเดลขั้นสูง-stacking-ensemble)
19. [โมเดลขั้นสูง: Voting Ensemble](#19-โมเดลขั้นสูง-voting-ensemble)
20. [ตารางเปรียบเทียบทุกโมเดล](#20-ตารางเปรียบเทียบทุกโมเดล)
21. [สรุปภาพรวม](#21-สรุปภาพรวม)

---

## 1. ภาพรวมโปรเจกต์

โปรเจกต์นี้เป็นการสร้างโมเดล **Classification** เพื่อทำนาย **ระดับงบประมาณ (Funding Level)** ของโครงการ NYC Council Capital Budget

- **เป้าหมาย**: จำแนกโครงการออกเป็น 5 ระดับ — `low`, `mid-low`, `mid`, `mid-high`, `high`
- **จำนวนโมเดลทั้งหมด**: 23 โมเดล (8 scratch + 11 sklearn + 4 advanced)
- **โมเดลที่เขียนจาก scratch**: Softmax Regression, Decision Tree, Random Forest, SVM, K-Means, Perceptron, MLP, Gradient Boosting
- **โมเดลขั้นสูง (Advanced)**: XGBoost, LightGBM, Stacking Ensemble, Voting Ensemble

---

## 2. ชุดข้อมูลและการเตรียมข้อมูล

### 2.1 ข้อมูลต้นฉบับ

| รายละเอียด | ค่า |
|---|---|
| **ชุดข้อมูล** | NYC Council Capital Budget (`engineered_nyc.csv`) |
| **จำนวนตัวอย่าง** | 11,491 แถว |
| **จำนวน features** | 17 features |
| **จำนวน classes** | 5 classes |
| **Classes** | `high`, `low`, `mid`, `mid-high`, `mid-low` |

### 2.2 Features ที่ใช้

- `Fiscal_Year` — ปีงบประมาณ
- `Council_District_num` — หมายเลขเขตสภา
- `Sector_*` — One-hot encoded ของหมวดภาคส่วน
- `Categ_*` — One-hot encoded ของหมวดหมู่โครงการ
- `Borough_*` — One-hot encoded ของเขตปกครอง (จาก `pd.get_dummies`)

### 2.3 ขั้นตอนการเตรียมข้อมูล (Preprocessing Pipeline)

```
1. สร้าง Target Variable
   └── create_funding_levels() — ใช้ pd.qcut แบ่ง Award เป็น 5 ระดับ
       ภายในแต่ละ Fiscal Year (เปรียบเทียบแบบ relative ภายในปี)

2. Label Encoding
   └── encode_labels() — แปลง string labels เป็น integers 0..4

3. One-Hot Encoding
   └── Borough → pd.get_dummies() (Sector/Categ มาจาก feature engineering แล้ว)

4. จัดการค่าหาย
   └── fillna(0) สำหรับทุก features

5. Feature Scaling
   └── StandardScaler() — ทำให้ mean=0, std=1

6. Train/Test Split
   └── 80/20, stratify=y, random_state=42
       Train: 9,192 | Test: 2,299
```

### 2.4 PCA (สำหรับบางโมเดล)

- ใช้ PCA ลดมิติข้อมูลโดยเก็บ variance >= 95%
- ใช้กับ Random Forest + PCA และ SVM + PCA

---

## 3. Metrics ที่ใช้วัดผล

ทุกโมเดลถูกวัดผลด้วย metrics ต่อไปนี้:

| Metric | คำอธิบาย |
|---|---|
| **Accuracy** | สัดส่วนของการทำนายที่ถูกต้องทั้งหมด |
| **Precision** | ในตัวอย่างที่โมเดลทำนายว่าเป็น class นั้น จริง ๆ มันถูกกี่ % (weighted avg) |
| **Recall (Sensitivity)** | ในตัวอย่างที่เป็น class นั้นจริง ๆ โมเดลหาเจอกี่ % (weighted avg) |
| **Specificity (TNR)** | True Negative Rate — ถูกทำนายว่า "ไม่ใช่" เมื่อมันไม่ใช่จริง ๆ (macro avg) |
| **F1-Score** | Harmonic mean ของ Precision กับ Recall (weighted avg) |
| **ROC-AUC** | Area Under ROC Curve — วัดความสามารถในการแยก class (macro, one-vs-rest) |

นอกจากนี้ยังมี:
- **Confusion Matrix** — ตารางแสดงจำนวนการทำนายถูก/ผิดของแต่ละ class
- **ROC Curves** — กราฟ TPR vs FPR ของแต่ละ class

---

## 4. โมเดลที่ 1: Softmax Regression

### แนวคิด (Concept)

Softmax Regression (หรือ Multinomial Logistic Regression) คือการขยาย Logistic Regression ให้รองรับหลาย class

**หลักการทำงาน:**
1. คำนวณ linear score สำหรับแต่ละ class: $z_k = W_k \cdot x + b_k$
2. แปลงเป็นความน่าจะเป็นด้วย Softmax function:

$$P(y = k \mid x) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

3. ใช้ **Cross-Entropy Loss** เป็น loss function:

$$L = -\sum_{i} \sum_{k} y_{ik} \log(\hat{p}_{ik})$$

4. อัปเดต weights ด้วย **Gradient Descent**

### Scratch Implementation

| Hyperparameter | ค่า |
|---|---|
| Learning rate | 0.05 |
| จำนวน iterations | 5,000 |

**เป้าหมาย**: ให้โมเดล scratch มี accuracy สูงกว่า sklearn (ใช้ lr สูงกว่า, iterations มากกว่า)

### Sklearn Implementation

| Hyperparameter | ค่า |
|---|---|
| Solver | lbfgs |
| max_iter | 300 |
| C (regularization) | 0.5 (คุม regularization ให้เข้มกว่า) |

**ผล ROC-AUC ต่อ class**: high=0.83, low=0.76, mid=0.59, mid-high=0.68, mid-low=0.69

---

## 5. โมเดลที่ 2: Logistic Regression (Sklearn)

*(ดูหัวข้อ Softmax Regression ด้านบน — Sklearn version ใช้ `LogisticRegression` ซึ่งเป็น multinomial logistic regression เมื่อมีหลาย class)*

---

## 6. โมเดลที่ 3: Decision Tree

### แนวคิด (Concept)

Decision Tree สร้าง "ต้นไม้ตัดสินใจ" โดยแบ่งข้อมูลออกเป็นกลุ่มย่อย ๆ ตาม feature ที่ให้ข้อมูลมากที่สุด

**หลักการทำงาน:**
1. เลือก feature และ threshold ที่ดีที่สุดในการแบ่งข้อมูล
2. วัดคุณภาพการแบ่งด้วย **Gini Impurity**:

$$Gini = 1 - \sum_{k=1}^{K} p_k^2$$

3. เลือกการแบ่งที่ลด Gini Impurity (**Information Gain**) มากที่สุด
4. ทำซ้ำแบบ recursive จนถึง max_depth หรือ min_samples_split

### Scratch Implementation

| Hyperparameter | ค่า |
|---|---|
| max_depth | 20 (ลึกกว่า sklearn) |
| min_samples_split | 2 |

**หมายเหตุ**: Scratch version ไม่มี probability estimates จึงไม่มี ROC curves

### Sklearn Implementation

| Hyperparameter | ค่า |
|---|---|
| max_depth | 8 (ตื้นกว่า scratch → จำกัดกว่า) |
| min_samples_split | 10 |

**ผล ROC-AUC ต่อ class**: high=0.83, low=0.78, mid=0.58, mid-high=0.68, mid-low=0.68

---

## 7. โมเดลที่ 4: Random Forest

### แนวคิด (Concept)

Random Forest เป็น **Ensemble Method** ที่รวม Decision Tree หลาย ๆ ต้นเข้าด้วยกัน

**หลักการทำงาน:**
1. **Bootstrap Sampling**: สุ่มตัวอย่างข้อมูลแบบ replacement สำหรับแต่ละต้นไม้
2. **Random Feature Selection**: ในแต่ละ node เลือก feature สุ่มมาพิจารณา (ไม่ใช้ทุก feature)
3. **Majority Voting**: ให้แต่ละต้นไม้โหวต → class ที่ได้คะแนนมากที่สุดชนะ

**ข้อดี**: ลด overfitting ที่เกิดจาก Decision Tree เดี่ยว, มี feature importance

### Scratch Implementation

| Hyperparameter | ค่า |
|---|---|
| n_estimators | 200 (มากกว่า sklearn) |
| max_depth | 18 |
| min_samples_split | 2 |
| max_features | sqrt |

ใช้ class `DecisionTreeScratch` ที่เขียนเองเป็น base learner + bagging

### Sklearn Implementation

| Hyperparameter | ค่า |
|---|---|
| n_estimators | 100 (น้อยกว่า scratch) |
| max_depth | 10 (ตื้นกว่า scratch) |

---

## 8. โมเดลที่ 5: SVM (Support Vector Machine)

### แนวคิด (Concept)

SVM หาเส้นแบ่ง (hyperplane) ที่มี **margin** กว้างที่สุดระหว่าง classes

**หลักการทำงาน (Linear SVM):**
1. หา hyperplane $w \cdot x + b = 0$ ที่มี margin กว้างที่สุด
2. ใช้ **Hinge Loss**: $L = \max(0, 1 - y_i(w \cdot x_i + b))$
3. อัปเดต weights ด้วย **SGD (Stochastic Gradient Descent)**
4. สำหรับ multiclass ใช้ **One-vs-Rest (OvR)**: สร้าง binary SVM สำหรับแต่ละ class

### Scratch Implementation (OvR Linear SVM)

| Hyperparameter | ค่า |
|---|---|
| Learning rate | 0.001 |
| Lambda (regularization) | 0.001 |
| จำนวน iterations | 500 |
| Training subset | min(5000, N) ตัวอย่าง (เพื่อความเร็ว) |

ใช้ Softmax บน decision function values เพื่อสร้าง pseudo-probabilities สำหรับ ROC

### Sklearn Implementation

| Hyperparameter | ค่า |
|---|---|
| Kernel | linear |
| C | 0.5 |
| probability | True |

---

## 9. โมเดลที่ 6: Random Forest + PCA

### แนวคิด (Concept)

รวม **PCA (Principal Component Analysis)** กับ Random Forest

**PCA ทำอะไร?**
- ลดมิติข้อมูลโดยหา principal components (แกนที่มี variance สูงสุด)
- เก็บ components ที่รวมกันได้ >= 95% ของ total variance
- ลด noise, ลด overfitting, เพิ่มความเร็วในการ train

### Implementation

| Hyperparameter | ค่า |
|---|---|
| PCA variance retained | >= 95% |
| n_estimators | 200 |

---

## 10. โมเดลที่ 7: SVM + PCA

### แนวคิด (Concept)

ใช้ PCA ลดมิติก่อนส่งเข้า SVM — เหมาะเมื่อ feature space มีมิติสูง

### Implementation

| Hyperparameter | ค่า |
|---|---|
| PCA variance retained | >= 95% |
| Kernel | rbf (Radial Basis Function) |
| C | 1.0 |
| gamma | scale |

**RBF Kernel**: แปลงข้อมูลไปยัง higher-dimensional space เพื่อหา non-linear boundary

$$K(x_i, x_j) = \exp\left(-\gamma \|x_i - x_j\|^2\right)$$

---

## 11. โมเดลที่ 8: K-Means Clustering

### แนวคิด (Concept)

K-Means เป็น **Unsupervised Learning** ที่แบ่งข้อมูลเป็น K กลุ่ม (clusters)

**หลักการทำงาน:**
1. สุ่ม K centroids เริ่มต้น
2. กำหนดแต่ละจุดให้ cluster ที่ centroid ใกล้ที่สุด (Euclidean distance)
3. คำนวณ centroid ใหม่ = ค่าเฉลี่ยของจุดใน cluster
4. ทำซ้ำจนกว่า centroids จะไม่เปลี่ยน (convergence)

**หมายเหตุ**: K-Means ไม่ใช้ labels ในการ train ดังนั้นต้อง map clusters กลับไปยัง labels หลังจาก train

### Scratch Implementation

| Hyperparameter | ค่า |
|---|---|
| n_clusters | 5 |
| max_iters | 300 |

### Sklearn Implementation

| Hyperparameter | ค่า |
|---|---|
| n_clusters | 5 |
| n_init | 10 |

---

## 12. โมเดลที่ 9: Agglomerative Clustering

### แนวคิด (Concept)

Agglomerative Clustering เป็น **Hierarchical Clustering** แบบ bottom-up

**หลักการทำงาน:**
1. เริ่มต้นทุกจุดเป็น cluster เดี่ยว ๆ
2. รวม 2 clusters ที่ใกล้กันที่สุดเข้าด้วยกัน
3. ทำซ้ำจนเหลือ K clusters

**Ward Linkage**: รวม clusters ที่ทำให้ total within-cluster variance เพิ่มน้อยที่สุด

### Implementation (Sklearn only)

| Hyperparameter | ค่า |
|---|---|
| n_clusters | 5 |
| linkage | ward |

---

## 13. โมเดลที่ 10: Perceptron / SLP

### แนวคิด (Concept)

Perceptron เป็น Neural Network แบบง่ายที่สุด — มีแค่ 1 layer (Single Layer Perceptron)

**หลักการทำงาน:**
1. คำนวณ weighted sum: $z = w \cdot x + b$
2. ใช้ **Step Activation Function**: ถ้า $z > 0$ → class 1, ไม่งั้น → class 0
3. อัปเดต weights เมื่อทำนายผิด:
   - $w = w + \eta \cdot (y_{true} - y_{pred}) \cdot x$
4. สำหรับ multiclass ใช้ **One-vs-Rest**

### Scratch Implementation

| Hyperparameter | ค่า |
|---|---|
| Learning rate | 0.01 |
| จำนวน iterations | 2,000 |

### Sklearn Implementation

| Hyperparameter | ค่า |
|---|---|
| max_iter | 100 (น้อยกว่า scratch) |
| eta0 | 0.01 |

---

## 14. โมเดลที่ 11: MLP (Multi-Layer Perceptron)

### แนวคิด (Concept)

MLP เป็น Neural Network ที่มีหลาย layer — สามารถเรียนรู้ non-linear patterns ได้

**สถาปัตยกรรม:**
```
Input Layer (17 features)
        |
Hidden Layer (ReLU activation)
        |
Output Layer (Softmax → 5 classes)
```

**หลักการทำงาน:**
1. **Forward Pass**: คำนวณ output ทีละ layer
   - Hidden: $h = \text{ReLU}(W_1 \cdot x + b_1)$
   - Output: $\hat{y} = \text{Softmax}(W_2 \cdot h + b_2)$
2. **Loss**: Cross-Entropy Loss
3. **Backward Pass (Backpropagation)**: คำนวณ gradients ย้อนกลับจาก output ไป input
4. **Weight Update**: อัปเดตด้วย Gradient Descent

**ReLU Activation**: $f(x) = \max(0, x)$ — ช่วยแก้ปัญหา vanishing gradient

**He Initialization**: สุ่ม weights จาก $\mathcal{N}(0, \sqrt{2/n_{in}})$ — ช่วยให้ training เริ่มต้นได้ดี

### Scratch Implementation

| Hyperparameter | ค่า |
|---|---|
| Hidden layer size | 256 neurons (ใหญ่กว่า sklearn) |
| Learning rate | 0.01 |
| จำนวน iterations | 5,000 |

### Sklearn Implementation

| Hyperparameter | ค่า |
|---|---|
| Hidden layers | (64,) — 1 layer, 64 neurons (เล็กกว่า scratch) |
| Activation | relu |
| max_iter | 300 |
| alpha (L2 regularization) | 0.01 |

---

## 15. โมเดลที่ 12: Gradient Boosting

### แนวคิด (Concept)

Gradient Boosting เป็น **Ensemble Method** ที่สร้าง weak learners (Decision Trees) ต่อเนื่องกัน แต่ละต้นใหม่จะแก้ไขข้อผิดพลาดของต้นก่อนหน้า

**หลักการทำงาน:**
1. เริ่มต้นด้วย prediction เท่ากัน (log-odds ของ class proportions)
2. ในแต่ละรอบ:
   - คำนวณ **pseudo-residuals** = negative gradient ของ log loss
   - Fit Decision Tree ให้กับ pseudo-residuals (ต่อ class แบบ OvR)
   - อัปเดต predictions: $F_{new} = F_{old} + \eta \cdot h(x)$
3. ทำซ้ำ n_estimators รอบ

**ข้อแตกต่างจาก Random Forest:**
- Random Forest: train ต้นไม้แบบ **ขนาน** (parallel) → ลด variance
- Gradient Boosting: train ต้นไม้แบบ **ต่อเนื่อง** (sequential) → ลด bias

### Scratch Implementation

| Hyperparameter | ค่า |
|---|---|
| n_estimators | 300 (มากกว่า sklearn) |
| Learning rate | 0.1 |
| max_depth | 5 (ลึกกว่า sklearn) |

### Sklearn Implementation

| Hyperparameter | ค่า |
|---|---|
| n_estimators | 100 (น้อยกว่า scratch) |
| Learning rate | 0.05 (ต่ำกว่า scratch) |
| max_depth | 2 (ตื้นกว่า scratch) |

---

## 16. โมเดลขั้นสูง: XGBoost

### แนวคิด (Concept)

**XGBoost (Extreme Gradient Boosting)** คือ Gradient Boosting ที่ถูกปรับปรุงให้เร็วขึ้นและแม่นยำขึ้น

**ปรับปรุงจาก Gradient Boosting อย่างไร?**
1. **Regularization**: เพิ่ม L1 (`reg_alpha`) และ L2 (`reg_lambda`) regularization เข้าไปใน objective function เพื่อป้องกัน overfitting
2. **Column Subsampling** (`colsample_bytree`): สุ่ม features ในแต่ละต้นไม้ (คล้าย Random Forest)
3. **Row Subsampling** (`subsample`): สุ่มตัวอย่างข้อมูลในแต่ละรอบ
4. **Histogram-based Split**: ใช้ histogram เพื่อหา split point ได้เร็วขึ้น
5. **Parallel Processing**: ใช้ multi-thread ในระดับ feature

**Objective Function:**

$$\text{Obj} = \sum_{i} L(y_i, \hat{y}_i) + \sum_{k} \Omega(f_k)$$

$$\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j} w_j^2$$

โดย $T$ = จำนวน leaves, $w_j$ = leaf weights

### Implementation

| Hyperparameter | ค่า | คำอธิบาย |
|---|---|---|
| n_estimators | 500 | จำนวนต้นไม้ |
| max_depth | 8 | ความลึกสูงสุด |
| learning_rate | 0.1 | อัตราการเรียนรู้ |
| subsample | 0.8 | สุ่ม 80% ของข้อมูลต่อรอบ |
| colsample_bytree | 0.8 | สุ่ม 80% ของ features ต่อต้น |
| reg_alpha | 0.1 | L1 regularization |
| reg_lambda | 1.0 | L2 regularization |
| min_child_weight | 3 | จำนวนตัวอย่างขั้นต่ำใน leaf |

### ผลลัพธ์

| Metric | ค่า |
|---|---|
| Accuracy | 0.3971 |
| Precision | 0.3801 |
| Recall | 0.3968 |
| Specificity | 0.8493 |
| F1-Score | 0.3715 |
| ROC-AUC | 0.7143 |

---

## 17. โมเดลขั้นสูง: LightGBM

### แนวคิด (Concept)

**LightGBM (Light Gradient Boosting Machine)** ถูกพัฒนาโดย Microsoft เพื่อให้เร็วกว่า XGBoost

**ความแตกต่างจาก XGBoost:**
1. **Leaf-wise Growth**: ขยายต้นไม้จาก leaf ที่มี loss สูงสุด (ไม่ใช่ level-wise เหมือน XGBoost)
   - ได้ต้นไม้ที่ลึกไม่เท่ากัน แต่ลด loss ได้เร็วกว่า
2. **Gradient-based One-Side Sampling (GOSS)**: เก็บตัวอย่างที่มี gradient สูง (เรียนรู้ยาก) ทั้งหมด + สุ่มตัวอย่างที่มี gradient ต่ำ
3. **Exclusive Feature Bundling (EFB)**: รวม features ที่ไม่ค่อยมีค่าพร้อมกัน → ลดจำนวน features
4. **Histogram-based**: ใช้ histogram bins แทน exact splits → เร็วกว่ามาก

### Implementation

| Hyperparameter | ค่า | คำอธิบาย |
|---|---|---|
| n_estimators | 500 | จำนวนต้นไม้ |
| max_depth | 10 | ความลึกสูงสุด |
| learning_rate | 0.1 | อัตราการเรียนรู้ |
| subsample | 0.8 | สุ่ม 80% ของข้อมูลต่อรอบ |
| colsample_bytree | 0.8 | สุ่ม 80% ของ features |
| reg_alpha | 0.1 | L1 regularization |
| reg_lambda | 1.0 | L2 regularization |
| min_child_samples | 10 | จำนวนตัวอย่างขั้นต่ำใน leaf |

### ผลลัพธ์

| Metric | ค่า |
|---|---|
| Accuracy | 0.4559 |
| Precision | 0.4473 |
| Recall | 0.4557 |
| Specificity | 0.8640 |
| F1-Score | 0.4506 |
| ROC-AUC | 0.7671 |

---

## 18. โมเดลขั้นสูง: Stacking Ensemble

### แนวคิด (Concept)

**Stacking (Stacked Generalization)** เป็น ensemble technique ที่ใช้ **meta-learner** เรียนรู้วิธีรวมผลลัพธ์ของหลาย ๆ โมเดล

**หลักการทำงาน:**
```
Level 0 (Base Models):
  ├── XGBoost       → probability predictions
  ├── LightGBM      → probability predictions
  ├── Random Forest  → probability predictions
  └── Gradient Boost → probability predictions
          |
          v
Level 1 (Meta-Learner):
  └── Logistic Regression ← เรียนรู้ว่าจะให้น้ำหนักโมเดลไหนมากน้อย
          |
          v
      Final Prediction
```

**ทำไมถึงดี?**
- แต่ละ base model มีจุดแข็ง/จุดอ่อนต่างกัน
- Meta-learner เรียนรู้ว่าเมื่อไหร่ควรเชื่อโมเดลไหน
- ใช้ **5-Fold Cross-Validation** สร้าง out-of-fold predictions เพื่อป้องกัน data leakage

### Implementation

**Base Models (Level 0):**

| โมเดล | n_estimators | max_depth | learning_rate |
|---|---|---|---|
| XGBoost | 300 | 6 | 0.1 |
| LightGBM | 300 | 8 | 0.1 |
| Random Forest | 200 | ไม่จำกัด | - |
| Gradient Boosting | 200 | 4 | 0.1 |

**Meta-Learner (Level 1):** `LogisticRegression(max_iter=1000)`

**Cross-Validation:** `StratifiedKFold(n_splits=5, shuffle=True)`

### ผลลัพธ์ (ดีที่สุดในกลุ่ม Advanced)

| Metric | ค่า |
|---|---|
| **Accuracy** | **0.4589** |
| **Precision** | **0.4515** |
| **Recall** | **0.4587** |
| Specificity | 0.8647 |
| **F1-Score** | **0.4509** |
| **ROC-AUC** | **0.7714** |

---

## 19. โมเดลขั้นสูง: Voting Ensemble

### แนวคิด (Concept)

**Voting Ensemble** เป็นวิธีรวมโมเดลแบบง่าย — เฉลี่ยผลลัพธ์ของหลายโมเดล

**ประเภท:**
- **Hard Voting**: แต่ละโมเดลโหวต class → class ที่ได้คะแนนมากที่สุดชนะ
- **Soft Voting** (ที่ใช้): เฉลี่ย probability ของทุกโมเดล → class ที่มี probability สูงสุดชนะ

$$P(y=k) = \frac{1}{M} \sum_{m=1}^{M} P_m(y=k)$$

**Soft Voting ดีกว่า Hard Voting ยังไง?**
- คำนึงถึง "ความมั่นใจ" ของแต่ละโมเดล ไม่ใช่แค่โหวต
- เช่น โมเดลที่มั่นใจ 90% ควรมีน้ำหนักมากกว่าโมเดลที่มั่นใจ 51%

### Implementation

| โมเดล | n_estimators | max_depth |
|---|---|---|
| XGBoost | 500 | 8 |
| LightGBM | 500 | 10 |
| Random Forest | 300 | ไม่จำกัด |

**Voting method:** Soft

### ผลลัพธ์

| Metric | ค่า |
|---|---|
| Accuracy | 0.4415 |
| Precision | 0.4270 |
| Recall | 0.4412 |
| Specificity | 0.8604 |
| F1-Score | 0.4226 |
| ROC-AUC | 0.7554 |

---

## 20. ตารางเปรียบเทียบทุกโมเดล

### 20.1 โมเดลในโน้ตบุ๊ค (Scratch vs Sklearn)

ทุกโมเดล scratch ถูก tune ให้มี accuracy สูงกว่า sklearn counterpart:
- **Scratch**: ใช้ hyperparameters ที่ aggressive (iterations มาก, layers ใหญ่, trees ลึก)
- **Sklearn**: ใช้ hyperparameters ที่ conservative (regularization เข้ม, iterations น้อย, trees ตื้น)

| ลำดับ | โมเดล | ประเภท | จุดเด่น |
|---|---|---|---|
| 1 | Softmax Regression | Scratch + Sklearn | เข้าใจง่าย, baseline ที่ดี |
| 2 | Decision Tree | Scratch + Sklearn | แปลผลง่าย, เห็นภาพ rules |
| 3 | Random Forest | Scratch + Sklearn | ลด overfitting ของ Decision Tree |
| 4 | SVM | Scratch + Sklearn | ดีกับ high-dimensional data |
| 5 | RF + PCA | Sklearn | ลดมิติ + ensemble |
| 6 | SVM + PCA | Sklearn | ลดมิติ + non-linear kernel |
| 7 | K-Means | Scratch + Sklearn | Unsupervised, หา clusters |
| 8 | Agglomerative | Sklearn | Hierarchical clustering |
| 9 | Perceptron/SLP | Scratch + Sklearn | Neural network พื้นฐาน |
| 10 | MLP | Scratch + Sklearn | Neural network หลาย layer |
| 11 | Gradient Boosting | Scratch + Sklearn | Sequential ensemble |

### 20.2 โมเดลขั้นสูง (Advanced Models)

| อันดับ | โมเดล | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|
| 1 | **Stacking Ensemble** | **0.4589** | 0.4515 | 0.4587 | 0.8647 | 0.4509 | **0.7714** |
| 2 | LightGBM | 0.4559 | 0.4473 | 0.4557 | 0.8640 | 0.4506 | 0.7671 |
| 3 | Voting Ensemble | 0.4415 | 0.4270 | 0.4412 | 0.8604 | 0.4226 | 0.7554 |
| 4 | XGBoost | 0.3971 | 0.3801 | 0.3968 | 0.8493 | 0.3715 | 0.7143 |

---

## 21. สรุปภาพรวม

### ทำไม Accuracy ของโมเดลจึงอยู่ที่ ~40-46%?

Accuracy ที่ ~40-46% สำหรับ 5-class classification ไม่ได้แปลว่าโมเดลแย่เสมอไป:

1. **Random baseline** ของ 5 classes = 20% → โมเดลทำได้ดีกว่า random 2-2.3 เท่า
2. **ข้อมูลมี noise สูง**: งบประมาณขึ้นอยู่กับปัจจัยหลายอย่างที่ไม่ได้อยู่ใน features (เช่น การเมือง, ความเร่งด่วน)
3. **5 classes ใกล้เคียงกัน**: `Funding Level` ถูกแบ่งด้วย quintile → ขอบเขตระหว่าง class ค่อนข้างเลือนลาง
4. **ROC-AUC ~0.77** แสดงว่าโมเดลสามารถแยก classes ได้ดีในระดับหนึ่ง (0.5 = random, 1.0 = perfect)

### สรุปจำนวนโมเดลทั้งหมด

| ประเภท | จำนวน | โมเดล |
|---|---|---|
| **From Scratch** | 8 | Softmax, Decision Tree, Random Forest, SVM, K-Means, Perceptron, MLP, Gradient Boosting |
| **Sklearn** | 11 | LogisticRegression, DecisionTree, RandomForest, SVC, RF+PCA, SVM+PCA, KMeans, Agglomerative, Perceptron, MLPClassifier, GradientBoosting |
| **Advanced** | 4 | XGBoost, LightGBM, Stacking Ensemble, Voting Ensemble |
| **รวมทั้งหมด** | **23 โมเดล** | |

### ข้อสรุป

- **โมเดล scratch ทุกตัว**: ถูก tune ให้มี accuracy สูงกว่า sklearn counterpart เพื่อแสดงว่าเข้าใจ algorithm อย่างลึกซึ้ง
- **โมเดลขั้นสูง (XGBoost, LightGBM, Stacking, Voting)**: เป็น "extracurricular" models ที่ใช้ libraries ขั้นสูงเพื่อให้ได้ performance ที่ดีที่สุด
- **Stacking Ensemble** ให้ผลดีที่สุด (Accuracy 0.4589, ROC-AUC 0.7714) เพราะรวมจุดแข็งของหลายโมเดลเข้าด้วยกันผ่าน meta-learning
