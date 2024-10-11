from flask import Flask, request, jsonify, session, redirect
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This will allow all origins by default
app.secret_key = 'your_secret_key'

users = {'admin': 'password123', 'user1': 'mypassword'}

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username'].strip()
    password = request.form['password'].strip()

    if username in users and users[username] == password:
        session['username'] = username
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'message': 'Invalid credentials'})

@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        return f"Welcome to your dashboard, {session['username']}!"
    else:
        return redirect("http://127.0.0.1:5500/website.html")

if __name__ == '__main__':
    app.run(debug=True, port=5000)
