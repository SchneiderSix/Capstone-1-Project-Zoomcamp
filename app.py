from flask import Flask, redirect, request, jsonify
from flasgger import Swagger
import tflite_runtime.interpreter as tflite
from PIL import Image
import time
import os
import requests
from datetime import datetime
import numpy as np

PORT = int(os.environ.get('PORT', 5000))


class TokenBucket:
    def __init__(self, capacity, refill_rate):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate  # Tokens per second
        self.last_refill = time.time()

    def refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        refill_amount = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + refill_amount)
        self.last_refill = now

    def take_token(self):
        self.refill()
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False


# Initialize the token bucket
bucket = TokenBucket(capacity=10, refill_rate=3)


def predict_wildfire(url):
    """Predicts a wildfire based on image

    Args:
      url (string): Image url

    Returns:
      float: Predicted value
    """

    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Create a unique filename using the current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"downloaded_image_{timestamp}.jpg"

        # Open a file in binary write mode and save the image
        with open(filename, "wb") as file:
            file.write(response.content)
        print(f"Image downloaded successfully as {filename}!")
    else:
        print("Failed to retrieve the image. Status code:", response.status_code)
        return

    interpreter = tflite.Interpreter(model_path='model.tflite')
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    preds = interpreter.get_tensor(output_index)

    with Image.open(filename) as img:
        img = img.resize((350, 350), Image.NEAREST)

    def preprocess_input(x):
        x /= 127.5
        x -= 1.
        return x

    x = np.array(img, dtype='float32')
    X = np.array([x])

    # Preprocess input
    X = preprocess_input(X)

    # Set tensor and invoke the interpreter after image preprocessing
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_index)

    classes = [
        'nowildfire',
        'wildfire'
    ]
    preds = preds[0].tolist()
    return dict(zip(classes, preds))


app = Flask(__name__)

# Initialize Swagger
swagger = Swagger(app)

# Rate limiter middleware


@app.before_request
def rate_limiter():
    if not bucket.take_token():
        return jsonify({"detail": "Rate limit exceeded"}), 429


# Define Flask routes
@app.route("/")
def index():
    # Redirect to the Swagger UI
    return redirect("/apidocs/")


@app.route("/predict", methods=['POST'])
def predict():
    """
    Predicts a wildfire based on image
    ---
    parameters:
      - name: query
        in: body
        required: true
        schema:
          type: object
          properties:
            query:
              type: string

    responses:
      200:
        description: Predicts a wildfire
        schema:
          type: object
          properties:
            answer:
              type: string
      400:
        description: Bad request due to missing query parameter
        schema:
          type: object
          properties:
            detail:
              type: string
      429:
        description: Rate limit exceeded
        schema:
          type: object
          properties:
            detail:
              type: string
      """

    data = request.json
    query = data.get('query')
    if not query:
        return jsonify({"detail": "Query parameter is required"}), 400
    try:
        answer = predict_wildfire(query)
        return ({"result": str(answer)}), 200
    except Exception as e:
        return jsonify({"detail": str(e)}), 500  # Handle unexpected errors


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=PORT)
