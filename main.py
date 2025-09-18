from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from fastapi.middleware.cors import CORSMiddleware

# ==== Load fine-tuned Vit5 model từ Hugging Face ====
MODEL_NAME = "lingling707/vit5-skinbot"  

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# ==== Pipeline chatbot ====
chatbot = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1  # Render không có GPU, dùng CPU
)

# ==== Model phân loại ảnh da ====
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


# ==== Mapping label -> skin type (tuỳ chỉnh theo nhu cầu) ====
def map_labels_to_skin_type(predictions):
    labels = [p["label"].lower() for p in predictions]
    if any(x in labels for x in ["acne", "pimple", "oily"]):
        return "da dầu, dễ nổi mụn"
    elif any(x in labels for x in ["wrinkle", "dry"]):
        return "da khô, có dấu hiệu lão hoá"
    elif any(x in labels for x in ["redness", "sensitive"]):
        return "da nhạy cảm"
    elif any(x in labels for x in ["blemish", "freckles"]):
        return "da hỗn hợp"
    elif "normal" in labels:
        return "da thường"
    else:
        return "không rõ, cần thêm ảnh chất lượng hơn"


# ==== Routes ====
@app.get("/")
async def root():
    return {"message": "Skin Advisor API (Vit5 + Skin Classification) is running 🚀"}


@app.post("/skinAdvisor")
async def skin_advisor(data: RequestData):
    skin_analysis = ""

    # Nếu có ảnh thì phân tích da
    if data.imageUrl and data.imageUrl.startswith(("http://", "https://")):
        try:
            results = image_model(data.imageUrl)
            skin_type = map_labels_to_skin_type(results)
            skin_analysis = f"Ảnh phân tích cho thấy: {skin_type}."
        except Exception as e:
            print("Image model error:", e)
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

    try:
        result = chatbot(prompt, max_new_tokens=120)
        reply = result[0]["generated_text"].strip()
        if not reply:
            reply = "Xin lỗi, mình chưa có câu trả lời phù hợp."
    except Exception as e:
        print("Chatbot error:", e)
        reply = "Xin lỗi, bot chưa trả lời được. Vui lòng thử lại."

    return {"reply": reply}
