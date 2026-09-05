const API = "http://127.0.0.1:8000";


/* ============================= */
/* LOAD EVERYTHING */
/* ============================= */

async function loadData() {

    try {

        await loadStats();

        await loadAttendance();

        await loadStudents();

        document.getElementById("status").innerText =
            "● System Online";

    }

    catch (error) {

        console.error(error);

        document.getElementById("status").innerText =
            "● Backend Offline";
    }
}


/* ============================= */
/* STATISTICS */
/* ============================= */

async function loadStats() {

    const response =
        await fetch(`${API}/stats/today`);

    const data =
        await response.json();


    document.getElementById(
        "totalStudents"
    ).innerText = data.total_students;


    document.getElementById(
        "totalAttendance"
    ).innerText = data.total_attendance;


    document.getElementById(
        "breakfast"
    ).innerText = data.breakfast;


    document.getElementById(
        "lunch"
    ).innerText = data.lunch;


    document.getElementById(
        "dinner"
    ).innerText = data.dinner;


    document.getElementById(
        "today"
    ).innerText = data.date;
}


/* ============================= */
/* ATTENDANCE */
/* ============================= */

async function loadAttendance() {

    const response =
        await fetch(
            `${API}/attendance/today`
        );

    const data =
        await response.json();


    const table =
        document.getElementById(
            "attendanceTable"
        );


    table.innerHTML = "";


    if (data.length === 0) {

        table.innerHTML = `
            <tr>
                <td colspan="6">
                    No attendance recorded today.
                </td>
            </tr>
        `;

        return;
    }


    data.forEach(record => {

        const row =
            document.createElement("tr");


        row.innerHTML = `

            <td>
                ${record.student_id}
            </td>

            <td>
                ${record.name || "Unknown"}
            </td>

            <td>
                ${record.roll_number || "-"}
            </td>

            <td>

                <span class="meal ${record.meal}">
                    ${record.meal}
                </span>

            </td>

            <td>
                ${record.time}
            </td>

            <td>
                ${(record.confidence * 100).toFixed(1)}%
            </td>

        `;


        table.appendChild(row);

    });
}


/* ============================= */
/* STUDENTS */
/* ============================= */

async function loadStudents() {

    const response =
        await fetch(
            `${API}/students`
        );

    const students =
        await response.json();


    const container =
        document.getElementById(
            "studentsContainer"
        );


    container.innerHTML = "";


    students.forEach(student => {

        const card =
            document.createElement("div");


        card.className =
            "student";


        card.innerHTML = `

            <div class="student-id">
                ${student.student_id}
            </div>

            <div class="student-name">
                ${student.name}
            </div>

            <div class="student-roll">
                Roll No: ${student.roll_number}
            </div>

        `;


        container.appendChild(card);

    });
}


/* ============================= */
/* INITIAL LOAD */
/* ============================= */

loadData();


/* ============================= */
/* AUTO REFRESH */
/* ============================= */

setInterval(
    loadData,
    5000
);