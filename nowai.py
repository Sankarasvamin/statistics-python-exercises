from fastapi import FastAPI
import pandas as pd

app = FastAPI()

@app.get("/")
def home():
    # 既然你脑子好、理解快，我们直接看这个：
    return {"status": "success", "msg": "Fudan Stats Student's API is LIVE"}

@app.get("/analyze")
def analyze_scores():
    # 模拟一份数据
    data = {'subject': ['Chemistry', 'Statistics'], 'score': [100, 95]}
    df = pd.DataFrame(data)
    
    return {
        "average": float(df['score'].mean()),
        "rank": "Specialist" if 100 in df.values else "Generalist"
    }