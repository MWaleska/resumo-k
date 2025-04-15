from flask import Flask, render_template, request
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import language_tool_python

app = Flask(__name__)

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

tool = language_tool_python.LanguageTool('pt-BR')

@app.route("/", methods=["GET", "POST"])
def index():
    resumo = ""
    if request.method == "POST":
        texto = request.form.get("texto", "")

        if texto.strip():
            try:
                matches = tool.check(texto)
                texto_corrigido = language_tool_python.utils.correct(texto, matches)

                inputs = tokenizer.encode(
                    texto_corrigido,
                    return_tensors="pt",
                    max_length=1024,
                    truncation=True
                ).to(device)

                summary_ids = model.generate(
                    inputs,
                    max_length=150,
                    min_length=40,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True
                )

                resumo = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

                matches_resumo = tool.check(resumo)
                resumo_corrigido = language_tool_python.utils.correct(resumo, matches_resumo)

                resumo = resumo_corrigido

            except Exception as e:
                resumo = f"Ocorreu um erro: {e}"

    return render_template("index.html", resumo=resumo)

if __name__ == "__main__":
    app.run(debug=True)