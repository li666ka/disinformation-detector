import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from datetime import datetime, timezone

BASE_PATH = os.path.join(os.path.dirname(__file__), "..")
DB_PATH = os.path.join(BASE_PATH, "diploma.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Experiment(Base):
    __tablename__ = "experiments"
    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(String, unique=True, index=True, nullable=False)
    user_id = Column(Integer, nullable=False)
    model_type = Column(String, nullable=False)
    model_file = Column(String, nullable=True)
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    train_size = Column(Integer, nullable=True)
    test_size = Column(Integer, nullable=True)
    training_time = Column(String, nullable=True)
    status = Column(String, default="success")
    # Model config details
    nb_variant = Column(String, nullable=True)      # multinomial | complement | bernoulli
    vectorizer = Column(String, nullable=True)       # tfidf | count
    xlm_mode = Column(String, nullable=True)         # concat | multiview
    llm_mode = Column(String, nullable=True)         # single | bagging
    # Ensemble
    ensemble_models = Column(Text, nullable=True)    # JSON: ["nb","deberta"]
    ensemble_strategy = Column(String, nullable=True)  # hard_voting | soft_voting | weighted_voting
    ensemble_weights = Column(Text, nullable=True)   # JSON: {"nb":0.3,"deberta":0.7}
    # v2: full model configs and feature groups
    model_configs = Column(Text, nullable=True)      # JSON: full ModelConfig[]
    feature_groups = Column(Text, nullable=True)     # JSON: ["emotional","stylistic"]
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class ModelRecord(Base):
    __tablename__ = "models"
    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(String, nullable=True)
    filename = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=True)
    model_path = Column(String, nullable=True)   # full path on Colab/Drive
    model_type = Column(String, nullable=False)
    accuracy = Column(Float, nullable=True)
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def create_tables():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
