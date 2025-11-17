from flask import Flask, request,jsonify
from db import DB

app = Flask(__name__)

# Preflight

@app.route("/users/<string:email>", methods=["OPTIONS"])
def pre_user_by_email(email):
    return '', 204, {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "DELETE",
        "Access-Control-Allow-Headers": "Content-Type"
    }


# Routes

# POST a user (expects form fields: name/username, email, password)
@app.route("/users", methods=["POST"])
def create_user():
    db = DB("Users.db")
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
