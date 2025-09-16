from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from fastapi.middleware.cors import CORSMiddleware
from datasets import load_dataset

# ==== Load fine-tune dataset (chỉ để demo, không train online) ====
# File: skincare_asking.jsonl, mỗi dòng {"input": "...", "output": "..."}
dataset_file = "skincare_asking.jsonl"
dataset = load_dataset("json", data_files=dataset_file)["train"]

# ==== Load tokenizer & model Vit5 fine-tune nếu có ====
# Nếu chưa fine-tune, dùng model base
model_name = "./vit5-skinbot"  # hoặc "VietAI/vit5-base" nếu chưa fine-tune
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# ==== Tạo pipeline chatbot ====
chatbot = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0  # GPU Colab: 0, CPU: -1
)

# ==== Model phân loại ảnh da ====
image_model = pipeline("image-classification", model="microsoft/resnet-50")

# ==== FastAPI setup ====
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==== Request body ====
class RequestData(BaseModel):
    userMessage: str
    imageUrl: Optional[str] = None

SKIN_LABELS = {"face","skin","pimple","freckles","wrinkle","acne","blemish","complexion","nose","cheek","forehead"}

def filter_skin_labels(predictions):
    return ", ".join([r['label'] for r in predictions if r['label'].lower() in SKIN_LABELS])

# ==== Routes ====
@app.get("/")
async def root():
    return {"message": "Skin Advisor API (Colab Fine-Tune Vit5) is running 🚀"}

@app.post("/skinAdvisor")
async def skin_advisor(data: RequestData):
    labels_text = ""
    if data.imageUrl and data.imageUrl.startswith(("http://","https://")):
        try:
            results = image_model(data.imageUrl)
            labels_text = filter_skin_labels(results)
        except:
            labels_text = ""

    # ==== Prompt cực an toàn, friendly ====
    if labels_text:
        prompt = (
            f"Bạn là chuyên gia tư vấn chăm sóc da. "
            f"Ảnh da người dùng cho thấy: {labels_text}. "
            f"Người dùng hỏi: {data.userMessage}. "
            "Trả lời ngắn gọn, thân thiện, bằng tiếng Việt, chỉ tư vấn chăm sóc da, tuyệt đối không gợi ý nguy hiểm."
        )
    else:
        prompt = (
            "Bạn là chuyên gia tư vấn chăm sóc da. "
            f"Người dùng hỏi: {data.userMessage}. "
            "Trả lời ngắn gọn, thân thiện, bằng tiếng Việt, chỉ tư vấn chăm sóc da, tuyệt đối không gợi ý nguy hiểm."
        )

    try:
        result = chatbot(
            prompt,
            max_new_tokens=80
        )
        # Loại bỏ prompt gốc nếu model trả về full text
        reply = result[0]["generated_text"].replace(prompt, "").strip()
    except:
        reply = "Xin lỗi, bot chưa trả lời được. Vui lòng thử lại."

    return {"reply": reply}