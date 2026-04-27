from __future__ import annotations

import json
from pathlib import Path
import sqlite3


class SQLiteRepository:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS requests (
                    request_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    channel TEXT,
                    symptoms TEXT,
                    image_filename TEXT,
                    status TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS predictions (
                    request_id TEXT PRIMARY KEY,
                    model_version TEXT NOT NULL,
                    top_label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    triage_label TEXT NOT NULL,
                    raw_scores TEXT NOT NULL,
                    latency_ms INTEGER NOT NULL,
                    FOREIGN KEY (request_id) REFERENCES requests(request_id)
                );

                CREATE TABLE IF NOT EXISTS errors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT,
                    error_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                );
                """
            )

    def create_request(
        self,
        request_id: str,
        timestamp: str,
        channel: str | None,
        symptoms: str | None,
        image_filename: str | None,
        status: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO requests (request_id, timestamp, channel, symptoms, image_filename, status)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (request_id, timestamp, channel, symptoms, image_filename, status),
            )

    def update_request_status(self, request_id: str, status: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE requests SET status = ? WHERE request_id = ?",
                (status, request_id),
            )

    def store_prediction(
        self,
        request_id: str,
        model_version: str,
        top_label: str,
        confidence: float,
        triage_label: str,
        raw_scores: dict[str, float],
        latency_ms: int,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO predictions
                (request_id, model_version, top_label, confidence, triage_label, raw_scores, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    model_version,
                    top_label,
                    confidence,
                    triage_label,
                    json.dumps(raw_scores),
                    latency_ms,
                ),
            )

    def store_error(
        self,
        request_id: str | None,
        error_type: str,
        message: str,
        timestamp: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO errors (request_id, error_type, message, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (request_id, error_type, message, timestamp),
            )

