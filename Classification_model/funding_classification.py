import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import PowerTransformer, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# หาตำแหน่งโฟลเดอร์โปรเจกต์ (โฟลเดอร์แม่ของ Classification_model)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "Data", "engineered_nyc.csv")


def parse_fiscal_year(x):
    """
    แปลงรูปแบบปีงบประมาณให้เป็นตัวเลข (เช่น 'FY19' -> 2019)
    ถ้าเป็นตัวเลขอยู่แล้วก็คืนค่าเดิม
    """
    x = str(x)
    if x.startswith("FY"):
        try:
            return 2000 + int(x[2:])
        except ValueError:
            return np.nan
    try:
        return int(x)
    except ValueError:
        return np.nan


def create_funding_levels(df, award_col="Award", year_col="Fiscal_Year"):
    """
    สร้าง label ระดับงบประมาณ (low, mid-low, mid, mid-high, high)
    โดยอ้างอิง distribution ของ Award ภายในปีงบประมาณเดียวกัน
    """
    labels = ["low", "mid-low", "mid", "mid-high", "high"]

    def _per_year(group):
        if group[award_col].nunique() < 5:
            # ถ้าปีนี้มีจำนวนน้อยหรือค่า Award ซ้ำมาก
            # จะใช้ qcut กับ rank เพื่อหลีกเลี่ยงปัญหา bin ซ้ำ
            ranks = group[award_col].rank(method="first")
            group["Funding_Level"] = pd.qcut(
                ranks,
                q=min(5, len(group)),
                labels=labels[: min(5, len(group))],
            )
        else:
            ranks = group[award_col].rank(method="first")
            group["Funding_Level"] = pd.qcut(
                ranks,
                q=5,
                labels=labels,
            )
        return group

    df = df.copy()
    # Preserve year_col: groupby can drop it in newer pandas
    saved_year = df[year_col].copy()
    df = df.groupby(year_col, group_keys=False).apply(_per_year)
    if year_col not in df.columns:
        df[year_col] = saved_year
    return df


def encode_labels(y):
    """
    แปลง Funding_Level จาก string เป็นตัวเลข 0..K-1
    """
    unique_labels = sorted(y.dropna().unique().tolist())
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    int_to_label = {idx: label for label, idx in label_to_int.items()}
    y_int = y.map(label_to_int).astype(int)
    return y_int, label_to_int, int_to_label


def select_features(df):
    """
    เลือกฟีเจอร์ที่ใช้สำหรับโมเดล
    ปรับชื่อคอลัมน์ตามไฟล์ engineered_nyc ของคุณหากแตกต่าง
    """
    df = df.copy()

    # Fix corrupted Council_District_num (NYC has 51 districts max)
    df["Council_District_num"] = df["Council_District_num"].clip(upper=51)

    # one-hot encoding สำหรับ Borough ถ้ายังไม่แปลง
    if not pd.api.types.is_numeric_dtype(df["Borough"]):
        borough_dummies = pd.get_dummies(df["Borough"], prefix="Borough")
        df = pd.concat([df, borough_dummies], axis=1)

    # เลือกคอลัมน์ตัวเลขและ one-hot (ปรับได้ตามต้องการ)
    feature_cols = [
        "Fiscal_Year",
        "Council_District_num",
    ]

    sector_cols = [c for c in df.columns if c.startswith("Sector_")]
    categ_cols = [c for c in df.columns if c.startswith("Categ_")]
    borough_cols = [c for c in df.columns if c.startswith("Borough_")]

    feature_cols.extend(sector_cols)
    feature_cols.extend(categ_cols)
    feature_cols.extend(borough_cols)

    # กรองเฉพาะคอลัมน์ที่มีอยู่จริง
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()

    # แทน missing ด้วย 0 (หรือจะใช้วิธีอื่นเช่น median ก็ได้)
    X = X.fillna(0)

    return X, feature_cols


class SoftmaxRegression:
    """
    Multiclass Logistic Regression (Softmax) แบบ from scratch ใช้ gradient descent
    """

    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.W = None
        self.b = None

    @staticmethod
    def _softmax(Z):
        Z_shift = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z_shift)
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    @staticmethod
    def _one_hot(y, num_classes):
        N = y.shape[0]
        one_hot = np.zeros((N, num_classes))
        one_hot[np.arange(N), y] = 1
        return one_hot

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        N, d = X.shape
        num_classes = len(np.unique(y))

        self.W = np.zeros((d, num_classes))
        self.b = np.zeros((1, num_classes))

        y_one_hot = self._one_hot(y, num_classes)

        for _ in range(self.n_iters):
            logits = X.dot(self.W) + self.b
            probs = self._softmax(logits)

            grad_logits = (probs - y_one_hot) / N
            grad_W = X.T.dot(grad_logits)
            grad_b = np.sum(grad_logits, axis=0, keepdims=True)

            self.W -= self.lr * grad_W
            self.b -= self.lr * grad_b

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        logits = X.dot(self.W) + self.b
        probs = self._softmax(logits)
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


def train_and_evaluate():
    """
    ฟังก์ชันหลักสำหรับรัน pipeline:
    - โหลดข้อมูล
    - สร้าง funding level
    - เตรียมฟีเจอร์
    - แบ่ง train/test
    - เทรน softmax regression from scratch
    - เทียบกับ sklearn LogisticRegression และ RandomForest
    - แสดงผลลัพธ์
    """
    print("อ่านข้อมูลจาก:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)

    # ลบแถวที่ไม่มีค่า Award หรือ Fiscal_Year
    df = df.dropna(subset=["Award", "Fiscal_Year"])

    # สร้าง label Funding_Level
    df = create_funding_levels(df, award_col="Award", year_col="Fiscal_Year")
    df = df.dropna(subset=["Funding_Level"])

    print("ตัวอย่าง Funding_Level:")
    print(df["Funding_Level"].value_counts())

    # เตรียมฟีเจอร์
    X, feature_cols = select_features(df)
    y_int, label_to_int, int_to_label = encode_labels(df["Funding_Level"])

    print("ใช้ฟีเจอร์ทั้งหมด:", len(feature_cols))
    print("คลาส:", label_to_int)

    # scaling ฟีเจอร์ (Yeo-Johnson: generalized Box-Cox that handles zeros)
    scaler = PowerTransformer(method='yeo-johnson', standardize=True)
    X_scaled = scaler.fit_transform(X.values)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y_int.values,
        test_size=0.2,
        random_state=42,
        stratify=y_int.values,
    )

    # -----------------------------
    # โมเดลจาก scratch: SoftmaxRegression
    # -----------------------------
    print("\nเทรนโมเดล SoftmaxRegression (from scratch)...")
    softmax_model = SoftmaxRegression(lr=0.01, n_iters=2000)
    softmax_model.fit(X_train, y_train)
    y_pred_softmax = softmax_model.predict(X_test)

    print("\n[SoftmaxRegression - From Scratch]")
    print("Accuracy:", accuracy_score(y_test, y_pred_softmax))
    print("Classification report:")
    print(
        classification_report(
            y_test, y_pred_softmax, target_names=[int_to_label[i] for i in sorted(int_to_label)]
        )
    )
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred_softmax))

    # -----------------------------
    # Sklearn: Logistic Regression
    # -----------------------------
    print("\nเทรนโมเดล LogisticRegression (sklearn)...")
    # รองรับ scikit-learn หลายเวอร์ชัน:
    # - บางเวอร์ชัน/บาง build อาจไม่รองรับพารามิเตอร์ multi_class
    # - ใช้ try/except แล้ว fallback ให้รันได้
    try:
        sk_logreg = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=1000,
        )
    except TypeError:
        sk_logreg = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
        )
    sk_logreg.fit(X_train, y_train)
    y_pred_sk = sk_logreg.predict(X_test)

    print("\n[LogisticRegression - Sklearn]")
    print("Accuracy:", accuracy_score(y_test, y_pred_sk))
    print("Classification report:")
    print(
        classification_report(
            y_test, y_pred_sk, target_names=[int_to_label[i] for i in sorted(int_to_label)]
        )
    )
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred_sk))

    # -----------------------------
    # Sklearn: Random Forest (ตัวอย่าง Ensemble)
    # -----------------------------
    print("\nเทรนโมเดล RandomForestClassifier (sklearn)...")
    rf_clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    rf_clf.fit(X_train, y_train)
    y_pred_rf = rf_clf.predict(X_test)

    print("\n[RandomForestClassifier - Sklearn]")
    print("Accuracy:", accuracy_score(y_test, y_pred_rf))
    print("Classification report:")
    print(
        classification_report(
            y_test, y_pred_rf, target_names=[int_to_label[i] for i in sorted(int_to_label)]
        )
    )
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred_rf))

    # -----------------------------
    # ตัวอย่าง ROC-AUC (macro) สำหรับ softmax model
    # -----------------------------
    try:
        num_classes = len(int_to_label)
        y_test_bin = label_binarize(y_test, classes=list(sorted(int_to_label)))
        y_proba_softmax = softmax_model.predict_proba(X_test)
        auc_macro = roc_auc_score(y_test_bin, y_proba_softmax, average="macro", multi_class="ovr")
        print("\nROC-AUC (macro) สำหรับ SoftmaxRegression:", auc_macro)
    except Exception as e:
        print("\nคำนวณ ROC-AUC ไม่สำเร็จ:", e)


if __name__ == "__main__":
    train_and_evaluate()

