"""SQLite storage for the history of saved frames and their detections."""
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS frames (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    ts_iso TEXT NOT NULL,
    source TEXT NOT NULL,
    model TEXT,
    width INTEGER, height INTEGER,
    n_det INTEGER DEFAULT 0,
    max_conf REAL DEFAULT 0,
    path TEXT NOT NULL,
    video_t REAL,
    flag TEXT,              -- NULL | 'ok' | 'missed'
    note TEXT
);
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    cls_id INTEGER, label TEXT, conf REAL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    wrong INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_det_frame ON detections(frame_id);
CREATE INDEX IF NOT EXISTS idx_frames_ts ON frames(ts);
"""


class DB:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ---- writes -------------------------------------------------
    def add_frame(self, ts, source, model, w, h, dets, path, video_t=None):
        with self.lock:
            cur = self.conn.execute(
                "INSERT INTO frames(ts,ts_iso,source,model,width,height,n_det,max_conf,path,video_t) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (ts, datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S"), source, model, w, h,
                 len(dets), max((d.conf for d in dets), default=0.0), str(path), video_t),
            )
            fid = cur.lastrowid
            self.conn.executemany(
                "INSERT INTO detections(frame_id,cls_id,label,conf,x1,y1,x2,y2) VALUES(?,?,?,?,?,?,?,?)",
                [(fid, d.cls_id, d.label, d.conf, d.x1, d.y1, d.x2, d.y2) for d in dets],
            )
            self.conn.commit()
            return fid

    def set_frame_flag(self, fid, flag, note=None):
        with self.lock:
            self.conn.execute("UPDATE frames SET flag=?, note=COALESCE(?, note) WHERE id=?", (flag, note, fid))
            self.conn.commit()

    def set_det_wrong(self, did, wrong):
        with self.lock:
            self.conn.execute("UPDATE detections SET wrong=? WHERE id=?", (1 if wrong else 0, did))
            self.conn.commit()

    def delete_frame(self, fid):
        with self.lock:
            row = self.conn.execute("SELECT path FROM frames WHERE id=?", (fid,)).fetchone()
            self.conn.execute("DELETE FROM detections WHERE frame_id=?", (fid,))
            self.conn.execute("DELETE FROM frames WHERE id=?", (fid,))
            self.conn.commit()
            return row["path"] if row else None

    # ---- reads --------------------------------------------------
    @staticmethod
    def _where(f):
        """Build a WHERE clause from a dict of filters."""
        cl, args = [], []
        if f.get("source"):
            cl.append("f.source=?"); args.append(f["source"])
        if f.get("model"):
            cl.append("f.model=?"); args.append(f["model"])
        if f.get("flag") == "wrong":
            cl.append("EXISTS(SELECT 1 FROM detections d WHERE d.frame_id=f.id AND d.wrong=1)")
        elif f.get("flag") == "none":
            cl.append("f.flag IS NULL AND NOT EXISTS(SELECT 1 FROM detections d WHERE d.frame_id=f.id AND d.wrong=1)")
        elif f.get("flag"):
            cl.append("f.flag=?"); args.append(f["flag"])
        if f.get("label") or f.get("min_conf"):
            sub = "EXISTS(SELECT 1 FROM detections d WHERE d.frame_id=f.id"
            if f.get("label"):
                sub += " AND d.label=?"; args.append(f["label"])
            if f.get("min_conf"):
                sub += " AND d.conf>=?"; args.append(float(f["min_conf"]))
            cl.append(sub + ")")
        if f.get("date_from"):
            cl.append("f.ts_iso>=?"); args.append(f["date_from"] + " 00:00:00")
        if f.get("date_to"):
            cl.append("f.ts_iso<=?"); args.append(f["date_to"] + " 23:59:59")
        return (" WHERE " + " AND ".join(cl)) if cl else "", args

    def _frames_with_dets(self, rows):
        out = []
        for r in rows:
            fr = dict(r)
            fr["detections"] = [dict(x) for x in self.conn.execute(
                "SELECT id,cls_id,label,conf,x1,y1,x2,y2,wrong FROM detections WHERE frame_id=? ORDER BY conf DESC", (fr["id"],))]
            out.append(fr)
        return out

    def history(self, filters, limit=40, offset=0):
        where, args = self._where(filters)
        with self.lock:
            total = self.conn.execute(f"SELECT COUNT(*) c FROM frames f{where}", args).fetchone()["c"]
            rows = self.conn.execute(
                f"SELECT f.* FROM frames f{where} ORDER BY f.ts DESC, f.id DESC LIMIT ? OFFSET ?", args + [limit, offset]).fetchall()
            return total, self._frames_with_dets(rows)

    def all_frames(self, filters):
        where, args = self._where(filters)
        with self.lock:
            rows = self.conn.execute(f"SELECT f.* FROM frames f{where} ORDER BY f.ts ASC", args).fetchall()
            return self._frames_with_dets(rows)

    def get_frame(self, fid):
        with self.lock:
            r = self.conn.execute("SELECT * FROM frames WHERE id=?", (fid,)).fetchone()
            return self._frames_with_dets([r])[0] if r else None

    def totals(self):
        with self.lock:
            c = self.conn
            return {
                "frames_saved": c.execute("SELECT COUNT(*) FROM frames").fetchone()[0],
                "detections_saved": c.execute("SELECT COUNT(*) FROM detections").fetchone()[0],
                "wrong_marked": c.execute("SELECT COUNT(*) FROM detections WHERE wrong=1").fetchone()[0],
                "missed_marked": c.execute("SELECT COUNT(*) FROM frames WHERE flag='missed'").fetchone()[0],
                "by_source": {r["source"]: r["n"] for r in c.execute("SELECT source, COUNT(*) n FROM frames GROUP BY source")},
                "by_class": [dict(r) for r in c.execute(
                    "SELECT label, COUNT(*) n, ROUND(AVG(conf),3) avg_conf FROM detections WHERE wrong=0 GROUP BY label ORDER BY n DESC")],
                "models": [r["model"] for r in c.execute("SELECT DISTINCT model FROM frames WHERE model IS NOT NULL")],
            }
