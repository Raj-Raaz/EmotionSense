from pydantic import BaseModel,Field

class EmotionRequest(BaseModel):
    text: str = Field(..., min_length=1, example="I am feeling very happy today!")

class EmotionResponse(BaseModel):
    emotion: str
    confidence: float