from fastapi import Query,FastAPI
from transformers import RobertaForSequenceClassification, AutoTokenizer
import torch
import uvicorn

app = FastAPI()

def sentiment_analyse(msg):
    model = RobertaForSequenceClassification.from_pretrained("wonrax/phobert-base-vietnamese-sentiment")

    tokenizer = AutoTokenizer.from_pretrained("wonrax/phobert-base-vietnamese-sentiment", use_fast=False)

    input_ids = torch.tensor([tokenizer.encode(msg)])

    with torch.no_grad():
        out = model(input_ids)
        result = out.logits.softmax(dim=-1).tolist()[0]  

        sentiments = ["Tiêu cực", "Tích cực", "Trung tính"]
        max_index = result.index(max(result))  
        max_sentiment = sentiments[max_index]  

        return {max_sentiment: result[max_index]}

@app.get("/api/sentiment/")
async def get_bot_response(msg: str = Query(...)):
    model = RobertaForSequenceClassification.from_pretrained("wonrax/phobert-base-vietnamese-sentiment")

    tokenizer = AutoTokenizer.from_pretrained("wonrax/phobert-base-vietnamese-sentiment", use_fast=False)

    input_ids = torch.tensor([tokenizer.encode(msg)])

    with torch.no_grad():
        out = model(input_ids)
        result = out.logits.softmax(dim=-1).tolist()[0]  

        sentiments = ["Tiêu cực", "Tích cực", "Trung tính"]
        max_index = result.index(max(result))  
        max_sentiment = sentiments[max_index]  

        return {max_sentiment: result[max_index]}
    # Output:
        # [[0.002, 0.988, 0.01]]
        #     ^      ^      ^
        #    NEG    POS    NEU
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)