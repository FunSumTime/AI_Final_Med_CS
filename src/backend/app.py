from flask import Flask, request,jsonify, g
from backend.db import DB
from backend.session_store import SessionStore
from executables.run import run_agent
import json

app = Flask(__name__)
session_store = SessionStore()

def load_session_data():
    auth_header = request.headers.get("Authorization")

    # default values so we don't get NameError
    sessionID = None
    session_data = None

    if auth_header and auth_header.startswith("Bearer "):
        sessionID = auth_header.removeprefix("Bearer ").strip()

    if sessionID is not None:
        session_data = session_store.get_session_data(sessionID)
        print("the session data is", session_data)

    if sessionID is None or session_data is None:
        # print("test")
        sessionID = session_store.create_session()
        session_data = session_store.get_session_data(sessionID)

    g.session_id = sessionID
    g.session_data = session_data


# Preflight

@app.before_request
def before_request_function():
    if request.method == "OPTIONS":
        response = app.response_class("", status=204)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return response
    load_session_data()

@app.after_request
def after_request_func(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

# @app.route("/users/<string:email>", methods=["OPTIONS"])
# def pre_user_by_email(email):
#     return '', 204, {
#         "Access-Control-Allow-Origin": "*",
#         "Access-Control-Allow-Methods": "DELETE",
#         "Access-Control-Allow-Headers": "Content-Type, Authorization"
#     }

@app.route("/sessions/settings",methods=["OPTIONS"])
def do_preflight():
    return '', 204, {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET,OPTIONS,PUT",
        "Access-Control-Allow-Headers": "Content-Type, Authorization"
    }

# @app.route("/sessions",methods=["OPTIONS"])
# def do_other_preflight():
#     return '', 204, {
#         "Access-Control-Allow-Origin": "*",
#         "Access-Control-Allow-Methods": "GET,OPTIONS,PUT,DELETE",
#         "Access-Control-Allow-Headers": "Content-Type, Authorization"
#     }

@app.route("/sessions/settings", methods=["PUT"])
def setsettings():
    load_session_data()
    data = request.form["data"]
    g.session_data["data"] = data
    # print(g.session_data)
    return "Data Saved", 200, {"Access-Control-Allow-Origin": "*"}

@app.route("/sessions", methods=["DELETE"])
def deleteSessionData():
    load_session_data
    session_store.delete_session(g.session_id)
    # print(g.session_data)
    return "Deleted", 200, {"Access-Control-Allow-Origin": "*"}

@app.route("/sessions", methods=["GET"])
def retrieveSession():
    load_session_data()
    return {
        "id": g.session_id,
        "data": g.session_data
    }, 200, {"Access-Control-Allow-Origin": "*"}

# Routes

# POST a user (expects form fields: name/username, email, password)
# give them a session
@app.route("/users", methods=["POST"])
def create_user():
    db = DB("Users.db")
    # might need to add firstname and last name
    d = {
        "username": request.form.get('username'),
        "email": request.form.get('email'),
        "password": request.form.get('password')
    }

    if not d["username"] or not d["email"] or not d["password"]:
        return "Missing required fields", 400, {"Access-Control-Allow-Origin": "*"}

    if db.user_exists(d["email"]):
        return "Email already exists", 409, {"Access-Control-Allow-Origin": "*"}

    try:
        uid = db.save_user(d) 
    except ValueError as e:
        return str(e), 400, {"Access-Control-Allow-Origin": "*"}

    return f"Created {uid}", 201, {"Access-Control-Allow-Origin": "*"}

# add authurization and sessions


# GET a user by id (no password returned)
@app.route("/users/<int:id>", methods=["GET"])
def get_user(id):
    db = DB("Users.db")
    user = db.get_user_by_id(id)
    if not user:
        return "Not found", 404, {"Access-Control-Allow-Origin": "*"}
    return user, 200, {"Access-Control-Allow-Origin": "*"}


# DELETE a user by email (also deletes related interactions via FK cascade)
@app.route("/users/<string:email>", methods=["DELETE"])
def delete_user(email):
    db = DB("Users.db")
    count = db.delete_user_by_email(email)
    if count == 0:
        return "Not found", 404, {"Access-Control-Allow-Origin": "*"}
    return "Deleted", 200, {"Access-Control-Allow-Origin": "*"}


@app.route("/users/login", methods=["POST"])
def check_user():
    d = {
        "email": request.form.get('email'),
        "password": request.form.get('password')
    }
    db = DB("Users.db")
    if not db.user_verify(d):
        return "Incorect", 401, {"Access-Control-Allow-Origin": "*"}

    return "Succesful retrieve", 200, {"Access-Control-Allow-Origin": "*"}

        

# post a users query
# POST a query/interaction (expects form: email, query, topic: 'CS' or 'MED')
@app.route("/interactions", methods=["POST"])
def post_interaction():
    db = DB("Users.db")
    d = {
        "email": request.form.get('email'),
        "query": request.form.get('query'),
        "topic": request.form.get('topic')  # 'CS' or 'MED'
    }
    if not d["email"] or not d["query"] or not d["topic"]:
        return "email, query, topic required", 400, {"Access-Control-Allow-Origin": "*"}
    if not db.user_exists(d["email"]):
        return "User not found", 404, {"Access-Control-Allow-Origin": "*"}

    try:
        interaction_id = db.log_interaction(d)
        
    except ValueError as e:
        return str(e), 400, {"Access-Control-Allow-Origin": "*"}
    agent_reply = None
    try:
        agent_reply = run_agent(email=d["email"],query=d["query"],topic=d["topic"], mode="chat")
    except Exception as e:
        print("Erorr in run_agent", e)
        agent_reply = None
    return jsonify({"status": "Logged", "interaction_id": interaction_id, "reply": agent_reply}), 201, {"Access-Control-Allow-Origin": "*"}


# route for the model
# matches: requests.get(.../interactions/history, data={"email":..., "limit":...})
@app.route("/interactions/history", methods=["GET"])
def interactions_history():
    db = DB("Users.db")
    email = request.form.get("email")
    limit = request.form.get("limit", 10)

    if not email:
        return "email is required", 400, {"Access-Control-Allow-Origin": "*"}

    try:
        limit = int(limit)
    except (ValueError, TypeError):
        limit = 10

    if not db.user_exists(email):
        return "User not found", 404, {"Access-Control-Allow-Origin": "*"}

    items = db.recent_interactions(email, limit=limit)
    return jsonify({"items": items}), 200, {"Access-Control-Allow-Origin": "*"}


# quizzes


@app.route("/quizzes", methods=["POST"])
def create_quiz():
    db = DB("Users.db")
    email = request.form.get("email")
    topic = request.form.get("topic")              # 'CS' or 'MED'
    quiz_request = request.form.get("quiz_request")
    include_history = request.form.get("include_history", "no")

    if not email or not topic or not quiz_request:
        return "Missing email, topic, or quiz_request", 400, {
            "Access-Control-Allow-Origin": "*"
        }

    if not db.user_exists(email):
        return "User not found", 404, {"Access-Control-Allow-Origin": "*"}

    # 1) Ask Jarvis to build the quiz (QUIZ MODE)
    try:
        raw_result = run_agent(
            email=email,
            topic=topic,
            mode="quiz",
            query=quiz_request,
            include_history=include_history,
        )
    except Exception as e:
        print("Error in run_agent (quiz):", e)
        return "Failed to generate quiz", 500, {"Access-Control-Allow-Origin": "*"}

    # raw_result might be:
    #  - a dict: {"quiz_json": "<json string>"}
    #  - or a raw JSON string wrapped in ``` fences (depending on how the model behaved)
    #  - or eventually a direct quiz object with "questions": [...]

    # Step 1: normalize to a Python object
    if isinstance(raw_result, str):
        cleaned = raw_result.strip()
        # strip ```json ... ``` if the model wrapped it
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            # remove leading "json" if present
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].lstrip()
        try:
            quiz_outer = json.loads(cleaned)
        except Exception as e:
            print("Could not json.loads raw_result:", e, raw_result)
            return "Agent returned non-JSON quiz", 500, {"Access-Control-Allow-Origin": "*"}
    else:
        quiz_outer = raw_result

    # Step 2: handle the two possible shapes:
    quiz_json_str = None

    if isinstance(quiz_outer, dict) and "quiz_json" in quiz_outer:
        # The shape you just showed in the logs
        quiz_json_str = quiz_outer["quiz_json"]

        # Optional: validate/normalize the inner JSON
        try:
            inner_quiz = json.loads(quiz_json_str)
        except Exception as e:
            print("quiz_json is not valid JSON:", e, quiz_json_str)
            return "quiz_json from agent is invalid", 500, {"Access-Control-Allow-Origin": "*"}

        # Normalize it just to be safe
        quiz_json_str = json.dumps(inner_quiz)

    elif isinstance(quiz_outer, dict) and "questions" in quiz_outer:
        # Future case: agent returns the quiz directly as an object
        quiz_json_str = json.dumps(quiz_outer)

    else:
        print("Agent returned unexpected quiz object:", quiz_outer)
        return "Agent did not return a valid quiz", 500, {"Access-Control-Allow-Origin": "*"}

    # 3) Save to DB
    quiz_id = db.save_quiz(email, topic, quiz_json_str)

    # 4) Return to frontend
    return jsonify({
        "quiz_id": quiz_id,
        "quiz_json": quiz_json_str
    }), 201, {"Access-Control-Allow-Origin": "*"}


@app.route("/quizzes/complete", methods=["POST"])
def complete_quiz():
    db = DB("Users.db")
    quiz_id = request.form.get("quiz_id")
    user_answers_json = request.form.get("user_answers_json")
    score = request.form.get("score")

    if not quiz_id or user_answers_json is None or score is None:
        return "Missing quiz_id, user_answers_json, or score", 400, {"Access-Control-Allow-Origin": "*"}

    try:
        qid = int(quiz_id)
        score_val = float(score)
    except ValueError:
        return "Invalid quiz_id or score", 400, {"Access-Control-Allow-Origin": "*"}

    db.mark_quiz_completed(qid, user_answers_json, score_val)
    return "Quiz completed", 200, {"Access-Control-Allow-Origin": "*"}

@app.route("/quizzes/history", methods=["POST"])
def quizzes_history():
    db = DB("Users.db")
    email = request.form.get("email")
    limit = request.form.get("limit", 5)

    if not email:
        return "email required", 400, {"Access-Control-Allow-Origin": "*"}

    try:
        limit = int(limit)
    except ValueError:
        limit = 5

    items = db.get_completed_quizzes_by_email(email, limit)
    print(items)
    return jsonify({"items": items}), 200, {"Access-Control-Allow-Origin": "*"}

@app.route("/quizzes/list", methods=["POST"])
def quizzes_list():
    db = DB("Users.db")
    email = request.form.get("email")
    limit = request.form.get("limit", 10)

    if not email:
        return "email required", 400, {"Access-Control-Allow-Origin": "*"}

    try:
        limit = int(limit)
    except ValueError:
        limit = 10

    items = db.get_quizzes_by_email(email, limit)
    return jsonify({"items": items}), 200, {"Access-Control-Allow-Origin": "*"}



@app.route("/home")
def welcome_home():
    return "<h1>Learning Coach API</h1>", 200, {"Access-Control-Allow-Origin": "*"}

def main():
    app.run(port=5000, debug=True)

if __name__ == "__main__":
    main()
