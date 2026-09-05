from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import sqlite3

import database


app = FastAPI(
    title="Mess Face Recognition API",
    version="1.0"
)
DATABASE = "mess.db"
# ==========================================
# STUDENT REGISTRATION MODEL
# ==========================================

class StudentRegistration(BaseModel):

    student_id: str
    name: str
    roll_number: str



# ==========================================
# STARTUP
# ==========================================

@app.on_event("startup")
def startup():

    database.initialize_database()


# ==========================================
# DATABASE HELPER
# ==========================================

def query_db(query, params=()):

    conn = sqlite3.connect(DATABASE)

    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    cursor.execute(query, params)

    rows = cursor.fetchall()

    conn.close()

    return [dict(row) for row in rows]


# ==========================================
# HOME
# ==========================================

@app.get("/")
def home():

    return {
        "system": "Mess Face Recognition",
        "status": "running"
    }

# ==========================================
# REGISTER STUDENT
# ==========================================

@app.post("/students/register")
def register_student(student: StudentRegistration):

    student_id = student.student_id.strip()
    name = student.name.strip()
    roll_number = student.roll_number.strip()

    if not student_id or not name or not roll_number:

        raise HTTPException(
            status_code=400,
            detail="All fields are required"
        )

    conn = sqlite3.connect(DATABASE)
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

        return {
            "success": True,
            "message": "Student registered successfully",
            "student": {
                "student_id": student_id,
                "name": name,
                "roll_number": roll_number
            }
        }

    except sqlite3.IntegrityError as e:

        if "student_id" in str(e):
            detail = "Student ID already exists"

        elif "roll_number" in str(e):
            detail = "Roll number already exists"

        else:
            detail = "Student already exists"

        raise HTTPException(
            status_code=409,
            detail=detail
        )

    finally:
        conn.close()


# ==========================================
# GET ALL STUDENTS
# ==========================================

@app.get("/students")
def get_students():
# ==========================================
# GET ALL STUDENTS
# ==========================================

@app.get("/students")
def get_students():

    return query_db("""
        SELECT
            student_id,
            name,
            roll_number
        FROM students
        ORDER BY student_id
    """)


# ==========================================
# GET SINGLE STUDENT
# ==========================================

@app.get("/students/{student_id}")
def get_student(student_id: str):

    result = query_db("""
        SELECT
            student_id,
            name,
            roll_number
        FROM students
        WHERE student_id = ?
    """, (student_id,))

    if not result:

        return {
            "error": "Student not found"
        }

    return result[0]


# ==========================================
# TODAY'S ATTENDANCE
# ==========================================

@app.get("/attendance/today")
def today_attendance():

    today = datetime.now().strftime(
        "%Y-%m-%d"
    )

    return query_db("""
        SELECT
            a.student_id,
            s.name,
            s.roll_number,
            a.meal,
            a.date,
            a.time,
            a.confidence
        FROM attendance a
        LEFT JOIN students s
        ON a.student_id = s.student_id
        WHERE a.date = ?
        ORDER BY a.time DESC
    """, (today,))


# ==========================================
# ATTENDANCE BY MEAL
# ==========================================

@app.get("/attendance/today/{meal}")
def meal_attendance(meal: str):

    today = datetime.now().strftime(
        "%Y-%m-%d"
    )

    return query_db("""
        SELECT
            a.student_id,
            s.name,
            s.roll_number,
            a.meal,
            a.date,
            a.time,
            a.confidence
        FROM attendance a
        LEFT JOIN students s
        ON a.student_id = s.student_id
        WHERE a.date = ?
        AND a.meal = ?
        ORDER BY a.time DESC
    """, (
        today,
        meal.lower()
    ))


# ==========================================
# TODAY'S STATISTICS
# ==========================================

@app.get("/stats/today")
def today_stats():

    today = datetime.now().strftime(
        "%Y-%m-%d"
    )

    total_students = query_db("""
        SELECT COUNT(*) AS count
        FROM students
    """)[0]["count"]

    total_attendance = query_db("""
        SELECT COUNT(*) AS count
        FROM attendance
        WHERE date = ?
    """, (today,))[0]["count"]

    breakfast = query_db("""
        SELECT COUNT(*) AS count
        FROM attendance
        WHERE date = ?
        AND meal = 'breakfast'
    """, (today,))[0]["count"]

    lunch = query_db("""
        SELECT COUNT(*) AS count
        FROM attendance
        WHERE date = ?
        AND meal = 'lunch'
    """, (today,))[0]["count"]

    dinner = query_db("""
        SELECT COUNT(*) AS count
        FROM attendance
        WHERE date = ?
        AND meal = 'dinner'
    """, (today,))[0]["count"]

    return {
        "date": today,
        "total_students": total_students,
        "total_attendance": total_attendance,
        "breakfast": breakfast,
        "lunch": lunch,
        "dinner": dinner
    }