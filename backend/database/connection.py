from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import redis
from decouple import config
import logging

logger = logging.getLogger(__name__)

DATABASE_URL = config("DATABASE_URL", default="sqlite:///./sports_predictor.db")
REDIS_HOST = config("REDIS_HOST", default="localhost")
REDIS_PORT = config("REDIS_PORT", default=6379, cast=int)

engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

redis_client = None

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_redis():
    global redis_client
    if redis_client is None:
        try:
            redis_client = redis.Redis(
                host=REDIS_HOST, 
                port=REDIS_PORT, 
                db=0, 
                decode_responses=True
            )
            redis_client.ping()
            logger.info("✅ Redis connected successfully")
        except redis.ConnectionError:
            logger.warning("❌ Redis not available, using in-memory cache")
            redis_client = SimpleCache()
    return redis_client

def init_db():
    from .models import Base
    Base.metadata.create_all(bind=engine)
    logger.info("✅ Database tables initialized")

class SimpleCache:
    def __init__(self):
        self.cache = {}
    
    def get(self, key):
        return self.cache.get(key)
    
    def set(self, key, value, ex=None):
        self.cache[key] = value
    
    def delete(self, key):
        if key in self.cache:
            del self.cache[key]
    
    def exists(self, key):
        return key in self.cache
    
    def ping(self):
        return True