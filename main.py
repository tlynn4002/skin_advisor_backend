# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
import requests
from transformers import pipeline

# ==== Hugging Face Inference API ====
MODEL_NAME = "lingling707/vit5-skinbot"
HF_API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"

# Nếu model public thì headers có thể để trống {}
# Nếu private thì tạo secret trên Render: HF_API_TOKEN
HF_HEADERS = {}

# ==== Model phân loại ảnh da (nhẹ, load local) ====
image_model = pipeline("image-classification", model="dima806/skin_types_image_detection")

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

# ==== Map label sang mô tả tiếng Việt ====
def map_labels_to_skin_type(label: str):
    label = label.lower()
    if label == "oily":
        return "da dầu, dễ nổi mụn"
    elif label == "dry":
        return "da khô, có thể bong tróc hoặc lão hóa sớm"
    elif label == "normal":
        return "da thường, cân bằng"
    else:
        return "không xác định rõ loại da"

# ==== Hàm gọi Hugging Face API ====
def query_hf_model(prompt: str):
    try:
        response = requests.post(
            HF_API_URL,
            headers=HF_HEADERS,
            json={"inputs": prompt}
        )
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
                return data[0]["generated_text"].strip()
            return "Xin lỗi, mình chưa có câu trả lời phù hợp."
        else:
            print("HF API error:", response.text)
            return "Xin lỗi, bot chưa trả lời được. Vui lòng thử lại."
    except Exception as e:
        print("HF API exception:", e)
        return "Có lỗi khi gọi Hugging Face API."

# ==== Routes ====
@app.get("/")
async def root():
    return {"message": "Skin Advisor API (Vit5 on Hugging Face API) is running 🚀"}

@app.post("/skinAdvisor")
async def skin_advisor(data: RequestData):
    skin_analysis = ""

    # Nếu có ảnh thì phân tích da
    if data.imageUrl and data.imageUrl.startswith(("http://","https://")):
        try:
            results = image_model(data.imageUrl)
            top = max(results, key=lambda x: x['score'])
            skin_type = map_labels_to_skin_type(top['label'])
            skin_analysis = f"Ảnh phân tích cho thấy: {skin_type}."
        except Exception as e:
            print("Image analysis error:", e)
            skin_analysis = "Không thể phân tích ảnh da."

    # Prompt chatbot
    if skin_analysis:
        prompt = (
            f"Bạn là chuyên gia tư vấn chăm sóc da. "
            f"{skin_analysis} "
            f"Người dùng hỏi: {data.userMessage}. "
            "Hãy trả lời ngắn gọn, thân thiện, bằng tiếng Việt, "
            "gồm cả nhận xét về tình trạng da và gợi ý phương hướng chăm sóc. "
            "Tuyệt đối không gợi ý nguy hiểm."
        )
    else:
        prompt = (
            "Bạn là chuyên gia tư vấn chăm sóc da. "
            f"Người dùng hỏi: {data.userMessage}. "
            "Hãy trả lời ngắn gọn, thân thiện, bằng tiếng Việt, "
            "chỉ tư vấn chăm sóc da, tuyệt đối không gợi ý nguy hiểm."
        )

    reply = query_hf_model(prompt)
    return {"reply": reply}
