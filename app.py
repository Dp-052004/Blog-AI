# #WORKING FLASK BACKEND
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# import torch
# from spellchecker import SpellChecker
# spell = SpellChecker()

# app = Flask(__name__)
# CORS(app)  # Enables CORS for all routes

# # Load the grammar correction model and tokenizer
# model_name = "prithivida/grammar_error_correcter_v1"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# @app.route('/correct', methods=['POST'])
# def correct_text():
#     data = request.get_json()
#     if not data or 'text' not in data:
#         return jsonify({"error": "Missing 'text' field"}), 400

#     prompt = f"gec: {data['text']}"
#     input_ids = tokenizer.encode(prompt, return_tensors="pt")

#     with torch.no_grad():
#         output = model.generate(input_ids, max_length=128)

#     corrected = tokenizer.decode(output[0], skip_special_tokens=True)
#     return jsonify({"corrected": corrected})
# @app.route('/spellcheck', methods=['POST'])
# def spellcheck_word():
#     try:
#         data = request.get_json()
#         if not data or 'word' not in data:
#             return jsonify({"error": "Missing 'word' field"}), 400

#         word = data['word']
        
#         # Check if the word is misspelled
#         misspelled = spell.unknown([word])
#         if not misspelled:
#             return jsonify({"suggestions": []})

#         # Get corrections (limit to 5 suggestions)
#         suggestions = spell.candidates(word)
#         suggestions = [str(s) for s in suggestions][:5]

#         return jsonify({"suggestions": suggestions})

#     except Exception as e:
#         print("Spellcheck Error:", e)
#         return jsonify({"error": "Internal server error"}),
# if __name__ == '__main__':
#     app.run(port=8001)


from flask import Flask, request, jsonify
from flask_cors import CORS
from spellchecker import SpellChecker

app = Flask(__name__)
CORS(app)

spell = SpellChecker()

# Cache variables for lazy loading
model = None
tokenizer = None

@app.route('/correct', methods=['POST'])
def correct_text():
    global model, tokenizer
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch

        if not model or not tokenizer:
            model_name = "prithivida/grammar_error_correcter_v1"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field"}), 400

        prompt = f"gec: {data['text']}"
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        with torch.no_grad():
            output = model.generate(input_ids, max_length=128)

        corrected = tokenizer.decode(output[0], skip_special_tokens=True)
        return jsonify({"corrected": corrected})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/spellcheck', methods=['POST'])
def spellcheck_word():
    try:
        data = request.get_json()
        if not data or 'word' not in data:
            return jsonify({"error": "Missing 'word' field"}), 400

        word = data['word']
        misspelled = spell.unknown([word])
        if not misspelled:
            return jsonify({"suggestions": []})

        suggestions = list(spell.candidates(word))[:5]
        return jsonify({"suggestions": suggestions})

    except Exception as e:
        return jsonify({"error": "Internal server error"}), 500




