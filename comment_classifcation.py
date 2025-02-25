import http.server
import socketserver
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB  # Use MultinomialNB for text data
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from urllib.parse import parse_qs
import traceback

PORT = 8001

try:
    dataset = pd.read_csv('train.csv')

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(dataset['comment_text'])  # Features
    Y = dataset["toxic"]  # Target

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=0)

    model = MultinomialNB()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    accuracy = accuracy_score(Y_test, Y_pred)
   
    print(f"Accuracy: {accuracy:.2f}")
except Exception as e:
    print(f"Error during model training: {e}")
    traceback.print_exc()

class MyHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            html_content = f"""

            <html>
                <head>
                    <title>Comment Toxicity Predictor</title>
                    <style>
                        body {{
                            font-family: 'Arial', sans-serif;
                            background-color: #f4f4f9;
                            color: #333;
                            margin: 0;
                            padding: 0;
                            display: flex;
                            justify-content: center;
                            align-items: center;
                            height: 100vh;
                            flex-direction: column;
                        }}
                        h1 {{
                            color: #4CAF50;
                            font-size: 36px;
                            margin-bottom: 20px;
                        }}
                        h2 {{
                            color: #333;
                            font-size: 28px;
                            margin-top: 40px;
                        }}
                        p {{
                            font-size: 18px;
                            line-height: 1.6;
                        }}
                        .container {{
                            background-color: #fff;
                            padding: 30px;
                            border-radius: 8px;
                            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                            width: 80%;
                            max-width: 800px;
                            text-align: center;
                        }}
                        form {{
                            margin-top: 20px;
                        }}
                        textarea {{
                            width: 100%;
                            padding: 10px;
                            font-size: 16px;
                            border-radius: 5px;
                            border: 1px solid #ccc;
                            margin-bottom: 20px;
                        }}
                        .btn {{
                            background-color: #4CAF50;
                            color: white;
                            padding: 10px 20px;
                            font-size: 18px;
                            border: none;
                            border-radius: 5px;
                            cursor: pointer;
                            transition: background-color 0.3s;
                        }}
                        .btn:hover {{
                            background-color: #45a049;
                        }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>Comment Toxicity Predictor</h1>
                        <p><strong>Model Accuracy:</strong> {accuracy:.2f}</p>

                        <h2>Enter a Comment:</h2>
                        <form method="POST" action="/predict">
                            <textarea name="comment" rows="4" placeholder="Type your comment here..."></textarea>
                            <br>
                            <button type="submit" class="btn">Predict Toxicity</button>
                        </form>
                    </div>
                </body>
            </html>
            """

            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()

            self.wfile.write(html_content.encode('utf-8'))
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == '/predict':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length).decode('utf-8')

                post_params = parse_qs(post_data)
                comment = post_params.get('comment', [''])[0]

                comment_vectorized = vectorizer.transform([comment])

                toxicity_level = model.predict(comment_vectorized)[0]

                html_content = f"""
                <html>
                    <head>
                        <title>Prediction Result</title>
                        <style>
                            body {{
                                font-family: 'Arial', sans-serif;
                                background-color: #f4f4f9;
                                color: #333;
                                margin: 0;
                                padding: 0;
                                display: flex;
                                justify-content: center;
                                align-items: center;
                                height: 100vh;
                                flex-direction: column;
                            }}
                            .container {{
                                background-color: #fff;
                                padding: 30px;
                                border-radius: 8px;
                                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                                width: 80%;
                                max-width: 800px;
                                text-align: center;
                            }}
                            .btn {{
                                background-color: #4CAF50;
                                color: white;
                                padding: 10px 20px;
                                font-size: 18px;
                                border: none;
                                border-radius: 5px;
                                cursor: pointer;
                                margin-top: 20px;
                                transition: background-color 0.3s;
                            }}
                            .btn:hover {{
                                background-color: #45a049;
                            }}
                        </style>
                    </head>
                    <body>
                        <div class="container">
                            <h1>Prediction Result</h1>
                            <p><strong>Comment:</strong> {comment}</p>
                            <p><strong>Toxicity Level:</strong> {'Toxic' if toxicity_level == 1 else 'Not Toxic'}</p>
                            <button class="btn" onclick="window.location.href='/'">Back to Home</button>
                        </div>
                    </body>
                </html>
                """

                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()

                self.wfile.write(html_content.encode('utf-8'))
            except Exception as e:
                print(f"Error during prediction: {e}")
                traceback.print_exc()
                self.send_error(500, "Internal Server Error")
        else:
            self.send_error(404, "File Not Found")

with socketserver.TCPServer(("", PORT), MyHandler) as httpd:
    print(f"Serving on port {PORT}...")
    httpd.serve_forever()
