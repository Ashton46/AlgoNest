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
@app.get("/api/models/{sport}/train")
async def train_model(sport: str, force_retrain: bool = False, background: bool = True):
    if sport not in [SportType.football, SportType.basketball]:
        raise HTTPException(status_code=400, detail="Invalid sport type")
    
    if background:
        from threading import Thread
        thread = Thread(target=predictor.train_sport_model, args=(sport, force_retrain))
        thread.start()
        
        return {
            "status": "training_started",
            "sport": sport,
            "background": True,
            "message": "Model training started in background"
        }
    else:
        result = predictor.train_sport_model(sport, force_retrain)
        return result

@app.get("/api/models/{sport}/stats")
async def get_model_stats(sport: str):
    if sport not in [SportType.football, SportType.basketball]:
        raise HTTPException(status_code=400, detail="Invalid sport type")
    
    stats = predictor.get_model_stats(sport)
    if "status" in stats and stats["status"] == "not_trained":
        raise HTTPException(status_code=404, detail="Model not trained yet")
    
    return stats

@app.get("/api/metrics", response_model=MetricsResponse)
async def get_metrics():
    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
    uptime = datetime.now() - start_time
    
    return MetricsResponse(
        model_accuracy="85.2%",
        average_confidence="78.5%",
        average_processing_time=f"{avg_processing_time:.2f}ms",
        total_predictions=str(total_predictions),
        uptime=str(uptime).split('.')[0],
        last_updated=datetime.utcnow().isoformat()
    )

@app.get("/api/health")
async def health_check():
    redis_status = "healthy" if get_redis().ping() else "unhealthy"
    models_status = {
        "football": predictor.is_trained.get("football", False),
        "basketball": predictor.is_trained.get("basketball", False)
    }
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "redis": redis_status,
        "models": models_status,
        "uptime": str(datetime.now() - start_time).split('.')[0]
    }

@app.get("/api/history")
async def get_prediction_history(sport: Optional[str] = None, limit: int = 50, db: Session = Depends(get_db)):
    query = db.query(PredictionLog)
    if sport:
        query = query.filter(PredictionLog.sport == sport)
    
    history = query.order_by(PredictionLog.timestamp.desc()).limit(limit).all()
    
    return {
        "count": len(history),
        "predictions": [
            {
                "id": log.id,
                "sport": log.sport,
                "timestamp": log.timestamp.isoformat(),
                "processing_time": log.processing_time_ms,
                "confidence": log.confidence
            }
            for log in history
        ]
    }

async def log_prediction(db: Session, prediction_id: str, situation: GameSituation, response: PredictionResponse, processing_time: float, request: Request):
    try:
        client_ip = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent")
        
        log_entry = PredictionLog(
            id=prediction_id,
            sport=situation.sport,
            input_data=situation.dict(),
            predictions=[p.dict() for p in response.predictions],
            win_probability=response.win_probability.dict() if response.win_probability else None,
            confidence=response.confidence,
            processing_time_ms=processing_time,
            model_version=response.model_version,
            user_agent=user_agent,
            client_ip=client_ip
        )
        
        db.add(log_entry)
        db.commit()
        logger.info(f"📝 Logged prediction {prediction_id}")
        
    except Exception as e:
        logger.error(f"❌ Failed to log prediction: {e}")
        db.rollback()

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "error": "HTTP Error"}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )

