import sqlite3
import json
import os
from datetime import datetime
from config import DATABASE_PATH

def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # User Accounts table (For Login / First-Time Signup)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            mobile TEXT NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Candidate details table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            aadhaar TEXT UNIQUE NOT NULL,
            aadhaar_photo_path TEXT,
            dob TEXT NOT NULL,
            gender TEXT NOT NULL,
            address TEXT NOT NULL,
            mobile TEXT NOT NULL,
            email TEXT NOT NULL,
            photo_path TEXT NOT NULL,
            face_embedding TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Ensure aadhaar_photo_path column exists in candidates table for existing databases
    cursor.execute("PRAGMA table_info(candidates)")
    columns = [row['name'] for row in cursor.fetchall()]
    if 'aadhaar_photo_path' not in columns:
        cursor.execute("ALTER TABLE candidates ADD COLUMN aadhaar_photo_path TEXT")
    
    # Driving Evaluation results table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_id TEXT NOT NULL,
            auth_status INTEGER NOT NULL,
            helmet_status INTEGER NOT NULL,
            lane_status INTEGER NOT NULL,
            foot_touch_status INTEGER NOT NULL,
            balance_status INTEGER NOT NULL,
            braking_status INTEGER DEFAULT 1,
            braking_state TEXT DEFAULT 'PASS',
            overall_result TEXT NOT NULL,
            rf_prediction TEXT,
            overall_score REAL NOT NULL,
            mistakes_json TEXT,
            video_path TEXT,
            processed_video_path TEXT,
            report_path TEXT,
            evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (candidate_id) REFERENCES candidates (candidate_id)
        )
    ''')

    # Ensure braking_status and braking_state columns exist for existing databases
    cursor.execute("PRAGMA table_info(evaluations)")
    eval_cols = [row['name'] for row in cursor.fetchall()]
    if 'braking_status' not in eval_cols:
        cursor.execute("ALTER TABLE evaluations ADD COLUMN braking_status INTEGER DEFAULT 1")
    if 'braking_state' not in eval_cols:
        cursor.execute("ALTER TABLE evaluations ADD COLUMN braking_state TEXT DEFAULT 'PASS'")
    
    conn.commit()
    conn.close()

def create_user(full_name, email, mobile, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO users (full_name, email, mobile, password)
            VALUES (?, ?, ?, ?)
        ''', (full_name, email, mobile, password))
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return True, user_id
    except sqlite3.IntegrityError as e:
        conn.close()
        return False, "An account with this Email ID already exists."

def authenticate_user(email, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE (email = ? OR mobile = ?) AND password = ?', (email, email, password))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def get_user(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def generate_candidate_id(year=None):
    if year is None:
        year = datetime.now().year
    prefix = f"CAND-{year}-"
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT candidate_id FROM candidates WHERE candidate_id LIKE ?', (f'{prefix}%',))
    rows = cursor.fetchall()
    
    max_num = 0
    for row in rows:
        cid = row['candidate_id']
        suffix = cid[len(prefix):]
        if suffix.isdigit():
            num = int(suffix)
            if num > max_num:
                max_num = num
                
    next_num = max_num + 1
    candidate_id = f"{prefix}{next_num:03d}"
    
    while True:
        cursor.execute('SELECT 1 FROM candidates WHERE candidate_id = ?', (candidate_id,))
        if not cursor.fetchone():
            break
        next_num += 1
        candidate_id = f"{prefix}{next_num:03d}"
        
    conn.close()
    return candidate_id

def register_candidate(candidate_data, face_embedding_list=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    embedding_json = json.dumps(face_embedding_list) if face_embedding_list is not None else None

    try:
        cursor.execute('''
            INSERT INTO candidates (
                candidate_id, name, aadhaar, aadhaar_photo_path, dob, gender, address, mobile, email, photo_path, face_embedding
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            candidate_data['candidate_id'],
            candidate_data['name'],
            candidate_data['aadhaar'],
            candidate_data.get('aadhaar_photo_path', ''),
            candidate_data['dob'],
            candidate_data['gender'],
            candidate_data['address'],
            candidate_data['mobile'],
            candidate_data['email'],
            candidate_data['photo_path'],
            embedding_json
        ))
        conn.commit()
        candidate_id = cursor.lastrowid
        conn.close()
        return True, candidate_id
    except sqlite3.IntegrityError as e:
        conn.close()
        return False, str(e)

def get_candidate(candidate_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM candidates WHERE candidate_id = ?', (candidate_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        c = dict(row)
        if c.get('face_embedding'):
            c['face_embedding'] = json.loads(c['face_embedding'])
        return c
    return None

def get_all_candidates():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM candidates ORDER BY created_at DESC')
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def save_evaluation(eval_data):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO evaluations (
            candidate_id, auth_status, helmet_status, lane_status, 
            foot_touch_status, balance_status, braking_status, braking_state,
            overall_result, rf_prediction, overall_score, mistakes_json, 
            video_path, processed_video_path, report_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        eval_data['candidate_id'],
        eval_data['auth_status'],
        eval_data['helmet_status'],
        eval_data['lane_status'],
        eval_data['foot_touch_status'],
        eval_data['balance_status'],
        eval_data.get('braking_status', 1),
        eval_data.get('braking_state', 'PASS'),
        eval_data['overall_result'],
        eval_data.get('rf_prediction', 'N/A'),
        eval_data['overall_score'],
        json.dumps(eval_data.get('mistakes', [])),
        eval_data.get('video_path', ''),
        eval_data.get('processed_video_path', ''),
        eval_data.get('report_path', '')
    ))
    conn.commit()
    eval_id = cursor.lastrowid
    conn.close()
    return eval_id

def get_evaluation(eval_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT e.*, c.name, c.aadhaar, c.dob, c.gender, c.mobile, c.email, c.photo_path
        FROM evaluations e 
        JOIN candidates c ON e.candidate_id = c.candidate_id 
        WHERE e.id = ?
    ''', (eval_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        res = dict(row)
        res['mistakes'] = json.loads(res['mistakes_json']) if res.get('mistakes_json') else []
        return res
    return None

def get_candidate_evaluations(candidate_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM evaluations WHERE candidate_id = ? ORDER BY evaluated_at DESC', (candidate_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

if __name__ == '__main__':
    init_db()
    print("Database initialized successfully.")
