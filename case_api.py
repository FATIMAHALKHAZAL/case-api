import joblib
from collections import Counter
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json

from fastapi import FastAPI
from pydantic import BaseModel

# =========================
# 1️⃣ تحميل الموديلات (نفس كودك بالضبط)
# =========================

# XGBoost + TF-IDF (Traineemodel1)
xgb_model = joblib.load("./Traineemodel1/xgb_model.pkl")
tfidf_1 = joblib.load("./Traineemodel1/tfidf_vectorizer.pkl")

# Logistic Regression + TF-IDF (Traineemodel3)
log_model = joblib.load("./Traineemodel3/logistic_model.pkl")
tfidf_3 = joblib.load("./Traineemodel3/tfidf_vectorizer.pkl")

# SVM + TF-IDF (Traineemodel4)
svm_model = joblib.load("./Traineemodel4/svm_model.pkl")
tfidf_4 = joblib.load("./Traineemodel4/tfidf_vectorizer.pkl")

# Naive Bayes + CountVectorizer (Traineemodel5)
nb_model = joblib.load("./Traineemodel5/case_classifier.pkl")
count_vec_5 = joblib.load("./Traineemodel5/count_vectorizer.pkl")

# BERT (Traineemodel2)
model_dir = "./Traineemodel2"

tokenizer = BertTokenizer.from_pretrained(model_dir)

bert_model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3
)
_ = bert_model.load_state_dict(torch.load(f"{model_dir}/bert_model_cases.bin", map_location="cpu"))
bert_model.eval()

with open(f"{model_dir}/label_map.json", "r", encoding="utf-8") as f:
    label_map = json.load(f)

# =========================
# 2️⃣ تحويل الرقم → اسم القسم (نفس فكرتك)
# =========================

def label_to_name(label_num: int) -> str:
    key = str(label_num)
    if key in label_map:
        return label_map[key]
    default = {
        0: "Personal Status",
        1: "Commercial",
        2: "General"
    }
    return default.get(label_num, f"Unknown ({label_num})")

# =========================
# 3️⃣ رأي الأغلبية
# =========================

def majority_vote(pred_list):
    return Counter(pred_list).most_common(1)[0][0]

# =========================
# 4️⃣ التنبؤ بكل الموديلات (نفس كودك)
# =========================

def predict_all_models(text: str):
    preds = {}

    # XGBoost
    preds["XGBoost"] = xgb_model.predict(tfidf_1.transform([text]))[0]

    # Logistic Regression
    preds["Logistic Regression"] = log_model.predict(tfidf_3.transform([text]))[0]

    # SVM
    svm_in = tfidf_4.transform([text]).toarray()
    preds["SVM"] = svm_model.predict(svm_in)[0]

    # Naive Bayes
    preds["Naive Bayes"] = nb_model.predict(count_vec_5.transform([text]))[0]

    # BERT
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=220,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = bert_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
    logits = outputs.logits
    bert_pred = torch.argmax(logits, dim=1).item()
    preds["BERT"] = bert_pred

    final_pred = majority_vote(list(preds.values()))
    return preds, final_pred

# =========================
# 5️⃣ دالة ترجع بس النتيجة النهائية كنص
# =========================

def get_final_label(text: str) -> str:
    _, final_answer = predict_all_models(text)
    final_label_name = label_to_name(final_answer)
    return final_label_name

# =========================
# 6️⃣ FastAPI – API جاهز للفويس فلو
# =========================

app = FastAPI()

class CaseRequest(BaseModel):
    text: str

class CaseResponse(BaseModel):
    prediction: str  # نوع القضية كنص

@app.post("/predict", response_model=CaseResponse)
def predict(req: CaseRequest):
    final_label_name = get_final_label(req.text)
    return {"prediction": final_label_name}
