# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
import requests
import os

# ==== Hugging Face Inference API (Vit5 chatbot) ====
HF_CHATBOT_MODEL = "lingling707/vit5-skinbot"
HF_CHATBOT_URL = f"https://api-inference.huggingface.co/models/{HF_CHATBOT_MODEL}"

# ==== Hugging Face Inference API (Skin classification) ====
HF_IMAGE_MODEL = "dima806/skin_types_image_detection"
HF_IMAGE_URL = f"https://api-inference.huggingface.co/models/{HF_IMAGE_MODEL}"

# Nếu model public thì headers có thể để {}
# Nếu private thì set token trong Render → Environment Variables: HF_API_TOKEN
HF_HEADERS = {}
if "HF_TOKEN" in os.environ:
    HF_HEADERS = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}

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
    if label in ["acne", "pimple", "oily"]:
        return "da dầu, dễ nổi mụn"
    elif label in ["dry", "wrinkle"]:
        return "da khô, có thể bong tróc hoặc lão hóa sớm"
    elif label in ["redness", "sensitive"]:
        return "da nhạy cảm"
    elif label in ["blemish", "freckles"]:
        return "da hỗn hợp"
    elif label == "normal":
        return "da thường, cân bằng"
    else:
        return "không xác định rõ loại da"

# ==== Hàm gọi HF API (chatbot) ====
def query_chatbot(prompt: str):
    try:
        response = requests.post(
            HF_CHATBOT_URL,
            headers=HF_HEADERS,
            json={"input": prompt}
        )
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
                return data[0]["generated_text"].strip()
            return "Xin lỗi, mình chưa có câu trả lời phù hợp."
        else:
            print("HF chatbot API error:", response.text)
            return "Xin lỗi, bot chưa trả lời được. Vui lòng thử lại."
    except Exception as e:
        print("HF chatbot API exception:", e)
        return "Có lỗi khi gọi Hugging Face API."

# ==== Hàm gọi HF API (image model) ====
def analyze_image(url: str):
    try:
        response = requests.post(
            HF_IMAGE_URL,
            headers=HF_HEADERS,
            json={"inputs": url}
        )
        if response.status_code == 200:
            results = response.json()
            if isinstance(results, list) and len(results) > 0:
                top = max(results, key=lambda x: x.get("score", 0))
                return top["label"]
        print("HF image API error:", response.text)
        return None
    except Exception as e:
        print("HF image API exception:", e)
        return None

# ==== Routes ====
@app.get("/")
async def root():
    return {"message": "Skin Advisor API (Vit5 + Skin Classification via HF API) is running 🚀"}

@app.post("/skinAdvisor")
async def skin_advisor(data: RequestData):
    skin_analysis = ""

    # Nếu có ảnh thì gọi API phân tích da
    if data.imageUrl and data.imageUrl.startswith(("http://", "https://")):
        label = analyze_image(data.imageUrl)
        if label:
            skin_type = map_labels_to_skin_type(label)
            skin_analysis = f"Ảnh phân tích cho thấy: {skin_type}."
        else:
            skin_analysis = "Không thể phân tích ảnh da."

    # Ghép prompt cho chatbot
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

    reply = query_chatbot(prompt)
    return {"reply": reply}
