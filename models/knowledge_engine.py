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
        self._migrate_schema()
        self._create_indexes()

    def _create_tables(self):
        self.cur.executescript("""
        CREATE TABLE IF NOT EXISTS research_runs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            status TEXT NOT NULL DEFAULT 'RUNNING',
            rows INTEGER,
            features INTEGER,
            notes TEXT,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            finished_at TIMESTAMP
        );

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
            generation INTEGER NOT NULL DEFAULT 0,
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
            generation INTEGER NOT NULL DEFAULT 0,
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

    def _column_names(self, table: str) -> set[str]:
        return {str(row[1]) for row in self.cur.execute(f"PRAGMA table_info({table})").fetchall()}

    def _table_exists(self, table: str) -> bool:
        row = self.cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        return row is not None

    def _ensure_column(self, table: str, column_sql: str, column_name: str) -> None:
        if self._table_exists(table) and column_name not in self._column_names(table):
            self.cur.execute(f"ALTER TABLE {table} ADD COLUMN {column_sql}")

    def _migrate_schema(self) -> None:
        self._ensure_column("experiments", "generation INTEGER NOT NULL DEFAULT 0", "generation")
        self._ensure_column("rankings", "generation INTEGER NOT NULL DEFAULT 0", "generation")
        self.conn.commit()

    def _create_indexes(self) -> None:
        if self._table_exists("experiments"):
            self.cur.execute("CREATE INDEX IF NOT EXISTS idx_experiments_run_generation ON experiments(run_id, generation)")
        if self._table_exists("rankings"):
            self.cur.execute("CREATE INDEX IF NOT EXISTS idx_rankings_run_generation ON rankings(run_id, generation)")
        self.conn.commit()

    def start_run(self, rows: int | None = None, features: int | None = None, notes: str | None = None) -> int:
        self.cur.execute(
            "INSERT INTO research_runs(status, rows, features, notes) VALUES ('RUNNING', ?, ?, ?)",
            (rows, features, notes),
        )
        self.conn.commit()
        return int(self.cur.lastrowid)

    def finish_run(self, run_id: int, status: str = "COMPLETED", notes: str | None = None) -> None:
        self.cur.execute(
            """UPDATE research_runs
               SET status = ?, notes = COALESCE(?, notes), finished_at = CURRENT_TIMESTAMP
               WHERE id = ?""",
            (status, notes, run_id),
        )
        self.conn.commit()

    def add_hypothesis(self, hypothesis):
        self._hypotheses.append((
            hypothesis.id,
            hypothesis.signature,
            hypothesis.direction,
            json.dumps([
                {"feature": c.feature, "operator": c.operator, "value": c.value}
                for c in hypothesis.conditions
            ]),
            len(hypothesis.conditions),
        ))

    def add_experiment(self, result: dict):
        self._experiments.append((
            result.get("run_id"),
            int(result.get("generation", 0)),
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
            result.get("runtime", 0.0),
        ))

    def add_ranking(self, run_id: int, hypothesis_id: str, rank: int, score: float, generation: int = 0):
        self._rankings.append((run_id, int(generation), hypothesis_id, rank, score))

    def flush(self):
        if self._hypotheses:
            self.cur.executemany(
                """INSERT OR IGNORE INTO hypotheses
                (id,signature,direction,conditions,feature_count)
                VALUES (?,?,?,?,?)""",
                self._hypotheses,
            )
        if self._experiments:
            self.cur.executemany(
                """INSERT INTO experiments(
                run_id,generation,hypothesis_id,
                train_win,validation_win,test_win,
                occurrence,expectancy,confidence,
                stability,gap,score,status,runtime)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                self._experiments,
            )
        if self._rankings:
            self.cur.executemany(
                """INSERT INTO rankings(run_id,generation,hypothesis_id,rank,score)
                VALUES (?,?,?,?,?)""",
                self._rankings,
            )
        self.conn.commit()
        self._hypotheses.clear()
        self._experiments.clear()
        self._rankings.clear()

    def top_rankings(self, limit=20, run_id: int | None = None, generation: int | None = None):
        sql = "SELECT * FROM rankings WHERE 1=1"
        params: list[object] = []
        if run_id is not None:
            sql += " AND run_id = ?"
            params.append(run_id)
        if generation is not None:
            sql += " AND generation = ?"
            params.append(generation)
        sql += " ORDER BY score DESC LIMIT ?"
        params.append(limit)
        return self.cur.execute(sql, tuple(params)).fetchall()

    def close(self):
        self.flush()
        self.conn.close()
