import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
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


class Dataset(Base):
    """
    User-uploaded dataset.

    Data layout on disk:
      uploaded_datasets/user_<user_id>/dataset_<id>/
        ├── manifest.json       — metadata (name, description, stats, components)
        ├── news.csv            — REQUIRED: article_id, article_text, article_label (FAKE|REAL), title, url, ...
        ├── tweets.csv          — OPTIONAL: tweet_id, article_id, text, user_id, ...
        ├── retweets.csv        — OPTIONAL: original_tweet_id, user_id
        ├── replies.csv         — OPTIONAL: reply_id, parent_tweet_id, parent_reply_id, text, user_id, ...
        ├── likes.csv           — OPTIONAL: tweet_id, user_id
        ├── users.csv           — OPTIONAL: user_id, screen_name, followers_count, ...
        └── evidence.csv        — OPTIONAL: article_id, url, content
    """
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Identification
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)

    # Storage path (absolute or project-relative folder)
    folder_path = Column(String, nullable=False)

    # news.csv stats (always computed at upload time)
    total_news = Column(Integer, default=0)
    fake_count = Column(Integer, default=0)
    real_count = Column(Integer, default=0)
    unlabeled_count = Column(Integer, default=0)  # rows with label=null

    # Which CSV files are present (detected at upload)
    has_news = Column(Boolean, default=True)       # always true if upload succeeded
    has_tweets = Column(Boolean, default=False)
    has_retweets = Column(Boolean, default=False)
    has_replies = Column(Boolean, default=False)
    has_likes = Column(Boolean, default=False)
    has_users = Column(Boolean, default=False)
    has_evidence = Column(Boolean, default=False)

    # Physical info
    file_size_bytes = Column(Integer, default=0)

    # State
    is_active = Column(Boolean, default=False)  # one per user can be active
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Experiment(Base):
    __tablename__ = "experiments"
    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(String, unique=True, index=True, nullable=False)
    user_id = Column(Integer, nullable=False)

    # Dataset used for training (NEW)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=True)

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
    nb_variant = Column(String, nullable=True)
    vectorizer = Column(String, nullable=True)
    xlm_mode = Column(String, nullable=True)
    llm_mode = Column(String, nullable=True)
    ensemble_models = Column(Text, nullable=True)
    ensemble_strategy = Column(String, nullable=True)
    ensemble_weights = Column(Text, nullable=True)
    model_configs = Column(Text, nullable=True)
    feature_groups = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class ModelRecord(Base):
    __tablename__ = "models"
    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(String, nullable=True)
    name = Column(String, nullable=False)
    model_type = Column(String, nullable=False)
    filename = Column(String, unique=True, nullable=True)
    model_path = Column(String, nullable=True)
    llm_config = Column(Text, nullable=True)
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
