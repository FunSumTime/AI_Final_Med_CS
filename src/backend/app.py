from flask import Flask, request,jsonify, g
from db import DB
from session_store import SessionStore

app = Flask(__name__)
session_store = SessionStore()

def load_session_data():
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        sessionID = auth_header.removeprefix("Bearer ")
    else:
        sessionID = None
    
    if sessionID:
        sesion_data = session_store.get_session_data(sessionID)
        print("the session data is", sesion_data)
    
    if sessionID == None or sesion_data == None:
        sessionID = session_store.create_session()
        sesion_data = session_store.get_session_data(sessionID)
    
    g.session_id = sessionID
    g.session_data = sesion_data


# Preflight

@app.route("/users/<string:email>", methods=["OPTIONS"])
def pre_user_by_email(email):
    return '', 204, {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "DELETE",
        "Access-Control-Allow-Headers": "Content-Type, Authorization"
    }

@app.route("/sessions/settings",methods=["OPTIONS"])
def do_preflight():
    return '', 204, {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET,OPTIONS,PUT",
        "Access-Control-Allow-Headers": "Content-Type, Authorization"
    }

@app.route("/sessions",methods=["OPTIONS"])
def do_other_preflight():
    return '', 204, {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET,OPTIONS,PUT,DELETE",
        "Access-Control-Allow-Headers": "Content-Type, Authorization"
    }

@app.route("/sessions/settings", methods=["PUT"])
def setsettings():
    load_session_data()
    data = request.form["data"]
    g.session_data["data"] = data
    return "Data Saved", 200, {"Access-Control-Allow-Origin": "*"}

@app.route("/sessions", methods=["DELETE"])
def deleteSessionData():
    load_session_data()
    if "data" in g.session_data:
        del g.session_data["data"]
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
        db.log_interaction(d)  
    except ValueError as e:
        return str(e), 400, {"Access-Control-Allow-Origin": "*"}

    return "Logged", 201, {"Access-Control-Allow-Origin": "*"}


# route for the model
@app.route("/interactions/history/", methods=["GET"])
def interactions_history():
    db = DB("Users.db")
    email = request.form.get("email")
    limit = request.form.get("limit", 10)
    if not db.user_exists(email):
        return "User not found", 404, {"Access-Control-Allow-Origin": "*"}
    items = db.recent_interactions(email, limit=limit)
    return jsonify({"items": items}), 200, {"Access-Control-Allow-Origin": "*"}

@app.route("/home")
def welcome_home():
    return "<h1>Learning Coach API</h1>", 200, {"Access-Control-Allow-Origin": "*"}

def main():
    app.run(port=5000, debug=True)

main()
