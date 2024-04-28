from src import api
from fastapi import FastAPI
import uvicorn

app = FastAPI()
app.include_router(api.router)

if __name__ == "__main__":    
    uvicorn.run(app, host="0.0.0.0", port=8000)