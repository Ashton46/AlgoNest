from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from datetime import datetime
import time
import logging
import uuid
from typing import Optional
from contextlib import asynccontextmanager

from models.schemas import GameSituation, PredictionResponse, SportType, MetricsResponse
from models.prediction_models import predictor
from database.connection import get_db, get_redis, init_db
from database.models import PredictionLog
from utils.kaggle_data import sports_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def startup_event(app: FastAPI):
    logger.info("Starting AI Sports Backend...")
    init_db()
    
    sports_data.initialize_kaggle()
    
    logger.info("Pre-training ML models...")
    
    football_result = predictor.train_sport_model("football")
    logger.info(f"🏈 Football model: {football_result.get('status', 'unknown')}")
    
    basketball_result = predictor.train_sport_model("basketball")
    logger.info(f"🏀 Basketball model: {basketball_result.get('status', 'unknown')}")
    
    logger.info("✅ Startup completed")
    yield
    logger.info("Application is shutting down.")

app = FastAPI(
    title="AI Sports Backend",
    description="Real-time sports predictions using Kaggle NFL & NBA datasets",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=startup_event
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

start_time = datetime.now()
total_predictions = 0
processing_times = []


@app.get("/")
async def root():
    return {
        "message": "AI Sports Backend API",
        "version": "2.0.0",
        "endpoints": {
            "docs": "/api/docs",
            "predict": "/api/predict",
            "health": "/api/health",
            "metrics": "/api/metrics",
            "train": "/api/models/{sport}/train"
        }
    }

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_play(
    situation: GameSituation,
    background_tasks: BackgroundTasks,
    request: Request,
    db: Session = Depends(get_db)
):
    start_time = time.time()
    prediction_id = str(uuid.uuid4())
    
    try:
        features = situation.dict()
        predictions = predictor.predict_plays(situation.sport, features)
        win_prob = predictor.get_win_probability(situation.sport, features)
        
        processing_time_ms = round((time.time() - start_time) * 1000, 2)
        
        global total_predictions, processing_times
        total_predictions += 1
        processing_times.append(processing_time_ms)
        if len(processing_times) > 1000:
            processing_times = processing_times[-1000:]
        
        response = PredictionResponse(
            predictions=predictions,
            win_probability=win_prob,
            confidence=round(max(p.probability for p in predictions), 3),
            processing_time=f"{processing_time_ms}ms",
            model_version="v2.0-kaggle",
            timestamp=datetime.utcnow().isoformat(),
            prediction_id=prediction_id
        )
        
        background_tasks.add_task(
            log_prediction,
            db, prediction_id, situation, response, processing_time_ms, request
        )
        
        redis_client = get_redis()
        if redis_client:
            cache_key = f"prediction:{prediction_id}"
            redis_client.set(cache_key, response.json(), ex=3600)
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

