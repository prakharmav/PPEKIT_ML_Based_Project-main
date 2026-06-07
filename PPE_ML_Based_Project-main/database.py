"""
SafeGuard AI — Incident Database Manager
SQLite-backed persistent storage for PPE violation logs and daily compliance stats.
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import os


class IncidentDatabase:
    def __init__(self, db_path: str = "safety_incidents.db"):
        self.db_path = db_path
        self._init_db()
        self.seed_database()

    # ------------------------------------------------------------------
    # Seed Database
    # ------------------------------------------------------------------
    def seed_database(self):
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM incidents")
            count = cur.fetchone()[0]
            if count > 0:
                return  # Already seeded
            
            import random
            from datetime import datetime, timedelta
            
            locations = ["Zone A", "Zone B", "Zone C"]
            violation_types = ["No Helmet", "Vest Missing", "No Helmet & Vest"]
            
            # Seed 14 days of data
            now = datetime.now()
            for d in range(14, -1, -1):
                target_date = now - timedelta(days=d)
                date_str = target_date.strftime("%Y-%m-%d")
                
                # Today gets exact events matching mockup
                if d == 0:
                    today_incidents = [
                        ("12:42:21", "No Helmet", 1, 1, 0.0, "incidents/dummy_no_helmet.jpg", "Zone A"),
                        ("12:42:10", "Vest Missing", 1, 1, 0.0, "incidents/dummy_no_vest.jpg", "Zone B"),
                        ("12:41:58", "No Helmet & Vest", 1, 1, 0.0, "incidents/dummy_no_helmet_vest.jpg", "Zone C")
                    ]
                    
                    total_w = 0
                    total_v = 0
                    compliances = []
                    
                    for t_str, v_type, w_c, v_c, comp_rate, s_path, loc in today_incidents:
                        timestamp = f"{date_str} {t_str}"
                        cur.execute("""
                            INSERT INTO incidents 
                            (timestamp, date, time, violation_type, worker_count, violation_count, compliance_rate, screenshot_path, location)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (timestamp, date_str, t_str, v_type, w_c, v_c, comp_rate, s_path, loc))
                        total_w += w_c
                        total_v += v_c
                        compliances.append(comp_rate)
                    
                    avg_c = sum(compliances) / len(compliances)
                    cur.execute("""
                        INSERT OR REPLACE INTO daily_stats (date, total_incidents, total_workers, avg_compliance)
                        VALUES (?, ?, ?, ?)
                    """, (date_str, len(today_incidents), total_w, avg_c))
                    continue

                # Historical random data
                num_incidents = random.randint(1, 4)
                daily_w = 0
                daily_v = 0
                compliances = []
                
                for idx in range(num_incidents):
                    h = random.randint(8, 17)
                    m = random.randint(0, 59)
                    s = random.randint(0, 59)
                    time_str = f"{h:02d}:{m:02d}:{s:02d}"
                    timestamp = f"{date_str} {time_str}"
                    
                    v_type = random.choice(violation_types)
                    loc = random.choice(locations)
                    
                    if v_type == "No Helmet":
                        s_path = "incidents/dummy_no_helmet.jpg"
                    elif v_type == "Vest Missing":
                        s_path = "incidents/dummy_no_vest.jpg"
                    else:
                        s_path = "incidents/dummy_no_helmet_vest.jpg"
                        
                    w_c = random.randint(1, 3)
                    v_c = random.randint(1, w_c)
                    comp_rate = round((w_c - v_c) / w_c * 100, 1)
                    
                    cur.execute("""
                        INSERT INTO incidents 
                        (timestamp, date, time, violation_type, worker_count, violation_count, compliance_rate, screenshot_path, location)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (timestamp, date_str, time_str, v_type, w_c, v_c, comp_rate, s_path, loc))
                    
                    daily_w += w_c
                    daily_v += v_c
                    compliances.append(comp_rate)
                
                avg_c = round(sum(compliances) / len(compliances), 1)
                cur.execute("""
                    INSERT OR REPLACE INTO daily_stats (date, total_incidents, total_workers, avg_compliance)
                    VALUES (?, ?, ?, ?)
                """, (date_str, num_incidents, daily_w, avg_c))
            
            conn.commit()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS incidents (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp        TEXT    NOT NULL,
                    date             TEXT    NOT NULL,
                    time             TEXT    NOT NULL,
                    violation_type   TEXT,
                    worker_count     INTEGER DEFAULT 0,
                    violation_count  INTEGER DEFAULT 0,
                    compliance_rate  REAL    DEFAULT 100.0,
                    screenshot_path  TEXT,
                    location         TEXT    DEFAULT 'Main Zone'
                );

                CREATE TABLE IF NOT EXISTS daily_stats (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    date            TEXT UNIQUE,
                    total_incidents INTEGER DEFAULT 0,
                    total_workers   INTEGER DEFAULT 0,
                    avg_compliance  REAL    DEFAULT 100.0
                );
            """)

    def _conn(self):
        return sqlite3.connect(self.db_path)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def log_incident(
        self,
        violation_type: str,
        worker_count: int,
        violation_count: int,
        screenshot_path: str = None,
        location: str = "Main Zone",
    ):
        now = datetime.now()
        ts   = now.strftime("%Y-%m-%d %H:%M:%S")
        date = now.strftime("%Y-%m-%d")
        time = now.strftime("%H:%M:%S")
        compliance = (
            (worker_count - violation_count) / worker_count * 100
            if worker_count > 0
            else 100.0
        )

        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO incidents
                    (timestamp, date, time, violation_type,
                     worker_count, violation_count, compliance_rate,
                     screenshot_path, location)
                VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (ts, date, time, violation_type,
                 worker_count, violation_count, compliance,
                 screenshot_path, location),
            )
            conn.execute(
                """
                INSERT INTO daily_stats (date, total_incidents, total_workers, avg_compliance)
                VALUES (?, 1, ?, ?)
                ON CONFLICT(date) DO UPDATE SET
                    total_incidents = total_incidents + 1,
                    total_workers   = total_workers + excluded.total_workers,
                    avg_compliance  = (avg_compliance * total_incidents + excluded.avg_compliance)
                                      / (total_incidents + 1)
                """,
                (date, worker_count, compliance),
            )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def get_all_incidents(self) -> pd.DataFrame:
        with self._conn() as conn:
            return pd.read_sql_query(
                "SELECT * FROM incidents ORDER BY timestamp DESC", conn
            )

    def get_recent_incidents(self, days: int = 7) -> pd.DataFrame:
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        with self._conn() as conn:
            return pd.read_sql_query(
                "SELECT * FROM incidents WHERE date >= ? ORDER BY timestamp DESC",
                conn, params=(cutoff,),
            )

    def get_daily_stats(self, days: int = 7) -> pd.DataFrame:
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        with self._conn() as conn:
            return pd.read_sql_query(
                "SELECT * FROM daily_stats WHERE date >= ? ORDER BY date",
                conn, params=(cutoff,),
            )

    def get_summary_stats(self) -> dict:
        today = datetime.now().strftime("%Y-%m-%d")
        with self._conn() as conn:
            cur = conn.cursor()

            cur.execute(
                "SELECT COUNT(*), COALESCE(SUM(worker_count),0), COALESCE(SUM(violation_count),0) "
                "FROM incidents WHERE date = ?",
                (today,),
            )
            t = cur.fetchone()

            cur.execute(
                "SELECT COUNT(*), COALESCE(SUM(worker_count),0), "
                "COALESCE(SUM(violation_count),0), COALESCE(AVG(compliance_rate),100.0) "
                "FROM incidents"
            )
            a = cur.fetchone()

        return {
            "today_incidents":  t[0],
            "today_workers":    t[1],
            "today_violations": t[2],
            "total_incidents":  a[0],
            "total_workers":    a[1],
            "total_violations": a[2],
            "avg_compliance":   round(a[3], 1),
        }

    def clear_all(self):
        with self._conn() as conn:
            conn.execute("DELETE FROM incidents")
            conn.execute("DELETE FROM daily_stats")
