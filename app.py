from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("advisor_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None
    advice = None

    if request.method == "POST":
        try:
            # Get form values
            cgpa = float(request.form["cgpa"])
            attendance = float(request.form["attendance"])
            study_hours = float(request.form["study_hours"])
            failed_courses = int(request.form["failed_courses"])
            assignment_score = float(request.form["assignment_score"])

            # Validation
            if not (0 <= cgpa <= 5):
                raise ValueError("CGPA must be between 0 and 5.")
            if not (0 <= attendance <= 100):
                raise ValueError("Attendance must be between 0 and 100.")
            if not (0 <= study_hours <= 24):
                raise ValueError("Study hours must be between 0 and 24.")
            if failed_courses < 0:
                raise ValueError("Failed courses cannot be negative.")
            if not (0 <= assignment_score <= 100):
                raise ValueError("Assignment score must be between 0 and 100.")

            # Prepare data for prediction
            student_data = [[
                cgpa,
                attendance,
                study_hours,
                failed_courses,
                assignment_score
            ]]

            # Make prediction
            prediction = model.predict(student_data)[0]
            result = str(prediction)

            # Generate simple advice based on the prediction
            advice = generate_advice(prediction, cgpa, attendance, study_hours, failed_courses, assignment_score)

        except ValueError as e:
            error = str(e)
        except Exception as e:
            error = f"An unexpected error occurred: {str(e)}"

    return render_template(
        "index.html",
        result=result,
        error=error,
        advice=advice
    )


def generate_advice(prediction, cgpa, attendance, study_hours, failed_courses, assignment_score):
    """Generate simple recommendations based on the prediction and input values"""
    advice_list = []

    prediction = str(prediction).lower()

    if "excellent" in prediction or "good" in prediction or prediction in ["A", "B", "High"]:
        advice_list.append("Great job! Keep maintaining your current study habits.")
        if study_hours < 3:
            advice_list.append("You can still improve further by increasing your daily study hours slightly.")
    else:
        advice_list.append("There's room for improvement. Focus on consistency.")

        if cgpa < 3.0:
            advice_list.append("Work on improving your CGPA by reviewing weak subjects regularly.")
        if attendance < 75:
            advice_list.append("Try to improve your class attendance — it has a strong impact on performance.")
        if study_hours < 4:
            advice_list.append("Increase your daily study hours. Aim for at least 4–5 focused hours.")
        if failed_courses > 0:
            advice_list.append(f"Focus on clearing the {failed_courses} failed course(s) as a priority.")
        if assignment_score < 60:
            advice_list.append("Improve your assignment performance — submit quality work on time.")

    if not advice_list:
        advice_list.append("Continue studying consistently and monitor your progress.")

    return advice_list


if __name__ == "__main__":
    app.run(debug=True)