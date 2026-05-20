import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker
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

    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)

    folder_path = Column(String, nullable=False)

    total_news = Column(Integer, default=0)
    fake_count = Column(Integer, default=0)
    real_count = Column(Integer, default=0)
    unlabeled_count = Column(Integer, default=0)

    has_news = Column(Boolean, default=True)
    has_tweets = Column(Boolean, default=False)
    has_retweets = Column(Boolean, default=False)
    has_replies = Column(Boolean, default=False)
    has_likes = Column(Boolean, default=False)
    has_users = Column(Boolean, default=False)
    has_evidence = Column(Boolean, default=False)

    file_size_bytes = Column(Integer, default=0)

    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    analytics_cache = Column(Text, nullable=True)

    active_split = Column(String, nullable=True, default=None)


class Experiment(Base):
    __tablename__ = "experiments"
    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(String, unique=True, index=True, nullable=False)
    user_id = Column(Integer, nullable=False)

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
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=True)
    name = Column(String, nullable=False)
    model_type = Column(String, nullable=False)
    pipeline_type = Column(String, nullable=False, default="tweet")
    filename = Column(String, unique=True, nullable=True)
    model_path = Column(String, nullable=True)
    llm_config = Column(Text, nullable=True)
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    metrics_json = Column(Text, nullable=True)
    predictions_json = Column(Text, nullable=True)
    is_active = Column(Boolean, default=False)
    splits_used = Column(String, nullable=True, default=None)
    inference_requirements = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Ensemble(Base):
    """Ансамбль натренованих моделей."""
    __tablename__ = "ensembles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String, nullable=False)

    voting_type = Column(String, nullable=False)
    member_model_ids = Column(Text, nullable=False)
    weights = Column(Text, nullable=True)

    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    f1_macro = Column(Float, nullable=True)
    roc_auc = Column(Float, nullable=True)
    metrics_json = Column(Text, nullable=True)

    splits_used = Column(String, nullable=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=True)

    is_active = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    user = relationship("User")
    dataset = relationship("Dataset")


def create_tables():
    Base.metadata.create_all(bind=engine)
    _run_sqlite_migrations()


def _run_sqlite_migrations():
    """Додає колонки, яких немає у вже існуючій SQLite БД (dev-режим)."""
    migrations = [
        ("datasets", "analytics_cache", "TEXT"),
        ("datasets", "active_split", "TEXT"),
        ("models", "precision", "REAL"),
        ("models", "recall", "REAL"),
        ("models", "f1_score", "REAL"),
        ("models", "metrics_json", "TEXT"),
        ("models", "pipeline_type", "TEXT DEFAULT 'tweet'"),
        ("models", "splits_used", "TEXT"),
        ("models", "dataset_id", "INTEGER"),
        ("models", "predictions_json", "TEXT"),
        ("models", "inference_requirements", "TEXT"),
    ]
    with engine.begin() as conn:
        for table, column, coltype in migrations:
            existing = {
                row[1] for row in conn.exec_driver_sql(f"PRAGMA table_info({table})")
            }
            if column not in existing:
                conn.exec_driver_sql(
                    f"ALTER TABLE {table} ADD COLUMN {column} {coltype}"
                )

        conn.exec_driver_sql("""
            CREATE TABLE IF NOT EXISTS ensembles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                voting_type TEXT NOT NULL,
                member_model_ids TEXT NOT NULL,
                weights TEXT,
                accuracy REAL,
                precision REAL,
                recall REAL,
                f1_score REAL,
                f1_macro REAL,
                roc_auc REAL,
                metrics_json TEXT,
                splits_used TEXT,
                dataset_id INTEGER,
                is_active BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (dataset_id) REFERENCES datasets(id)
            )
        """)
        conn.exec_driver_sql("""
            CREATE INDEX IF NOT EXISTS idx_ensembles_user
            ON ensembles(user_id, is_active)
        """)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
