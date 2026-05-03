from flask import Flask, render_template, request, url_for
import joblib
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

model = joblib.load("fast_fake_news_model.pkl")

def generate_wordcloud(text, bg_color, colormap):
    wc = WordCloud(width=600, height=400, background_color=bg_color, colormap=colormap).generate(text)
    buf = BytesIO()
    wc.to_image().save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def generate_bar(pred):
    fig, ax = plt.subplots(figsize=(5,1.5))
    labels = ["Real", "Fake"]
    values = [1,0] if pred==0 else [0,1]
    colors = ["#0d6efd", "#dc3545"]
    ax.barh(labels, values, color=colors, height=0.6)
    ax.set_xlim(0,1)
    ax.set_xticks([])
    ax.set_title("Prediction Result", pad=10)
    for i, v in enumerate(values):
        ax.text(v + 0.02, i, labels[i], color=colors[i], fontweight='bold', va='center')
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', transparent=True)
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    title_input = ""
    text_input = ""
    fake_wc = None
    real_wc = None
    label_chart = None

    if request.method == "POST":
        title_input = request.form.get("news_title", "")
        text_input = request.form.get("news_text", "")
        combined_text = title_input + " " + text_input

        pred = model.predict([combined_text])[0]
        prediction = "Fake News " if pred == 1 else "Real News "

        label_chart = generate_bar(pred)

        if pred == 1:
            fake_wc = generate_wordcloud(combined_text, "#f8d7da", "Reds")
        else:
            real_wc = generate_wordcloud(combined_text, "#d1ecf1", "Blues")

    return render_template(
        "home.html",
        prediction=prediction,
        title_input=title_input,
        text_input=text_input,
        fake_wc=fake_wc,
        real_wc=real_wc,
        label_chart=label_chart
    )

if __name__ == "__main__":
    app.run(debug=True)

