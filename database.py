import sqlite3
from datetime import datetime

DATABASE = "mess.db"


def get_connection():
    return sqlite3.connect(DATABASE)


def initialize_database():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            student_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            roll_number TEXT UNIQUE NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            meal TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            confidence REAL,
            UNIQUE(student_id, meal, date),
            FOREIGN KEY(student_id)
                REFERENCES students(student_id)
        )
    """)

    conn.commit()
    conn.close()


def add_student(student_id, name, roll_number):

    conn = get_connection()
    cursor = conn.cursor()

    try:

        cursor.execute("""
            INSERT INTO students
            (student_id, name, roll_number)
            VALUES (?, ?, ?)
        """, (
            student_id,
            name,
            roll_number
        ))

        conn.commit()

        print(f"Student added: {student_id}")

    except sqlite3.IntegrityError as e:

        print(f"Could not add student: {e}")

    conn.close()


def get_student(student_id):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT student_id, name, roll_number
        FROM students
        WHERE student_id = ?
    """, (student_id,))

    student = cursor.fetchone()

    conn.close()

    return student


def already_marked(student_id, meal):

    today = datetime.now().strftime("%Y-%m-%d")

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id
        FROM attendance
        WHERE student_id = ?
        AND meal = ?
        AND date = ?
    """, (
        student_id,
        meal,
        today
    ))

    result = cursor.fetchone()

    conn.close()

    return result is not None


def mark_attendance(
    student_id,
    meal,
    confidence
):

    now = datetime.now()

    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    conn = get_connection()
    cursor = conn.cursor()

    try:

        cursor.execute("""
            INSERT INTO attendance
            (student_id, meal, date, time, confidence)
            VALUES (?, ?, ?, ?, ?)
        """, (
            student_id,
            meal,
            date,
            time,
            confidence
        ))

        conn.commit()

        print(
            f"✅ Attendance marked: "
            f"{student_id} | {meal}"
        )

        result = True

    except sqlite3.IntegrityError:

        print(
            f"⚠ Already marked: "
            f"{student_id} | {meal}"
        )

        result = False

    conn.close()

    return result


if __name__ == "__main__":

    initialize_database()

    print("Database initialized.")