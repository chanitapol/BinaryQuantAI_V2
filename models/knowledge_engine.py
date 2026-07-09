from __future__ import annotations

import json
import sqlite3
from pathlib import Path


class KnowledgeEngine:
    def __init__(self, db_path: str = "research.db"):
        self.conn = sqlite3.connect(Path(db_path))
        self.conn.row_factory = sqlite3.Row
        self.cur = self.conn.cursor()

        self._hypotheses = []
        self._experiments = []
        self._rankings = []

        self._create_tables()

    def _create_tables(self):
        self.cur.executescript("""
        CREATE TABLE IF NOT EXISTS hypotheses(
            id TEXT PRIMARY KEY,
            signature TEXT UNIQUE,
            direction TEXT,
            conditions TEXT,
            feature_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS experiments(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            hypothesis_id TEXT,
            train_win REAL,
            validation_win REAL,
            test_win REAL,
            occurrence INTEGER,
            expectancy REAL,
            confidence REAL,
            stability REAL,
            gap REAL,
            score REAL,
            status TEXT,
            runtime REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS rankings(
            run_id INTEGER,
            hypothesis_id TEXT,
            rank INTEGER,
            score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS metadata(
            key TEXT PRIMARY KEY,
            value TEXT
        );
        """)
        self.conn.commit()

    def add_hypothesis(self, hypothesis):
        self._hypotheses.append((
            hypothesis.id,
            hypothesis.signature,
            hypothesis.direction,
            json.dumps([
                {
                    "feature": c.feature,
                    "operator": c.operator,
                    "value": c.value
                }
                for c in hypothesis.conditions
            ]),
            len(hypothesis.conditions)
        ))

    def add_experiment(self, result: dict):
        self._experiments.append((
            result.get("run_id"),
            result["hypothesis_id"],
            result["train_winrate"],
            result["validation_winrate"],
            result["test_winrate"],
            result["occurrence"],
            result["expectancy"],
            result["confidence"],
            result["stability"],
            result["gap"],
            result["score"],
            result["status"],
            result.get("runtime", 0.0)
        ))

    def add_ranking(self, run_id: int, hypothesis_id: str, rank: int, score: float):
        self._rankings.append((
            run_id,
            hypothesis_id,
            rank,
            score
        ))

    def flush(self):
        if self._hypotheses:
            self.cur.executemany(
                """INSERT OR IGNORE INTO hypotheses
                (id,signature,direction,conditions,feature_count)
                VALUES (?,?,?,?,?)""",
                self._hypotheses
            )

        if self._experiments:
            self.cur.executemany(
                """INSERT INTO experiments(
                run_id,hypothesis_id,
                train_win,validation_win,test_win,
                occurrence,expectancy,confidence,
                stability,gap,score,status,runtime)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                self._experiments
            )

        if self._rankings:
            self.cur.executemany(
                """INSERT INTO rankings(
                run_id,hypothesis_id,rank,score)
                VALUES (?,?,?,?)""",
                self._rankings
            )

        self.conn.commit()

        self._hypotheses.clear()
        self._experiments.clear()
        self._rankings.clear()

    def top_rankings(self, limit=20):
        return self.cur.execute(
            """SELECT * FROM rankings
            ORDER BY score DESC
            LIMIT ?""",
            (limit,)
        ).fetchall()

    def close(self):
        self.flush()
        self.conn.close()
