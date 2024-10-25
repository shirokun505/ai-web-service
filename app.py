from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline, set_seed

app = Flask(__name__)
CORS(app)

# Load the model (this will happen on cold start)
generator = pipeline('text-generation', model='mak109/distilgpt2-finetuned-lyrics', framework='tf')
set_seed(42)

@app.route('/api/generate_lyrics', methods=['POST'])
def generate_lyrics():
    data = request.json
    prompt = data.get('prompt')
    language = data.get('language')
    genre = data.get('genre')

    if not prompt or not language or not genre:
        return jsonify({"error": "Missing prompt, language, or genre"}), 400

    try:
        full_prompt = f"Generate lyrics for a {genre} song in {language}. The song is about: {prompt}\n\nLyrics:\n"
        response = generator(full_prompt, max_length=200, num_return_sequences=1, temperature=0.7)
        lyrics = response[0]['generated_text'].strip().split("Lyrics:\n")[-1].strip()
        return jsonify({"lyrics": lyrics})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# This is necessary for local development
if __name__ == '__main__':
    app.run(debug=True)
