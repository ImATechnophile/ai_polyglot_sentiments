import gradio as gr
import whisper
from transformers import pipeline

model = whisper.load_model("base")
sentiment_analysis = pipeline("sentiment-analysis", framework="pt", model="SamLowe/roberta-base-go_emotions")

def analyze_sentiment(text):
    results = sentiment_analysis(text)
    sentiment_results = {result['label']: result['score'] for result in results}
    return sentiment_results

def get_sentiment_emoji(sentiment):
    emoji_mapping = {
        "disappointment": "😞", "sadness": "😢", "annoyance": "😠", "neutral": "😐", "disapproval": "👎",
        "realization": "😮", "nervousness": "😬", "approval": "👍", "joy": "😄", "anger": "😡",
        "embarrassment": "😳", "caring": "🤗", "remorse": "😔", "disgust": "🤢", "grief": "😥",
        "confusion": "😕", "relief": "😌", "desire": "😍", "admiration": "😌", "optimism": "😊",
        "fear": "😨", "love": "❤️", "excitement": "🎉", "curiosity": "🤔", "amusement": "😄",
        "surprise": "😲", "gratitude": "🙏", "pride": "🦁"
    }
    return emoji_mapping.get(sentiment, "")

def display_sentiment_results(sentiment_results, option):
    sentiment_text = ""
    for sentiment, score in sentiment_results.items():
        emoji = get_sentiment_emoji(sentiment)
        if option == "Sentiment Only":
            sentiment_text += f"{sentiment} {emoji}\n"
        elif option == "Sentiment + Score":
            sentiment_text += f"{sentiment} {emoji}: {round(score, 2)}\n"  # Rounded to 2 decimal places
    return sentiment_text

def inference(audio, sentiment_option):
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    _, probs = model.detect_language(mel)
    lang = max(probs, key=probs.get)

    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)

    sentiment_results = analyze_sentiment(result.text)
    sentiment_output = display_sentiment_results(sentiment_results, sentiment_option)

    return lang.upper(), result.text, sentiment_output

# Updated UI for a more modern and user-friendly experience
title = """<h1 align="center">🌍 PolyGlot Sentiments 💭</h1>"""
image_path = "assets/logo_1.png"

description = """
<div style="text-align: center; font-size: 16px; padding: 15px; background-color: #f0f8ff; border-radius: 15px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);">
    <p>
        <strong>🚀 This demo leverages <span style="color: #007bff;">Whisper</span> for real-time <u>multilingual speech transcription</u></strong> 
        <br>
        and the <strong style="color: #ff6347;">RoBERTa Base model</strong> for advanced <u>sentiment analysis</u>.
    </p>
    <p>
        <span style="color: #28a745; font-size: 18px;">🌟 Get instant transcription insights with sentiment analysis,</span>
        <br>
        <strong>presented through fun emojis that reflect emotions!</strong> 🎉
    </p>
</div>
"""

custom_css = """
#banner-image {
    display: block;
    margin-left: auto;
    margin-right: auto;
    width: 200px; /* Increased image size */
}
#chat-message {
    font-size: 14px;
    min-height: 300px;
    border: 1px solid #e0e0e0;
    padding: 10px;
    border-radius: 10px;
    background-color: #f9f9f9;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
.footer {
    font-size: 12px;
    text-align: center;
    padding-top: 20px;
}
#centered-radio {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    margin: 0 auto;
}
.transcribe-btn {
    background-color: #007bff !important; /* Changed button color to blue */
    border-color: #007bff !important;
}
"""

block = gr.Blocks(css=custom_css)

with block:
    gr.HTML(title)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Image(image_path, elem_id="banner-image", show_label=False)

    with gr.Group():
        gr.HTML(description)

        with gr.Row():
            audio = gr.Audio(
                label="Upload or Record Audio",
                show_label=True,
                type="filepath"
            )

            sentiment_option = gr.Radio(
                choices=["Sentiment Only", "Sentiment + Score"],
                label="Select Sentiment Display Option",
                value="Sentiment Only",
                info="Choose how to display sentiment analysis results.",
                elem_id="centered-radio"
            )

        with gr.Row():
            btn = gr.Button("Transcribe & Analyze", variant="primary", elem_classes="transcribe-btn")

        with gr.Row():
            lang_str = gr.Textbox(label="Detected Language", interactive=False)

            text = gr.Textbox(label="Transcription", interactive=False)

        with gr.Row():
            sentiment_output = gr.Textbox(label="Sentiment Analysis Results", interactive=False, elem_id="chat-message")

        btn.click(inference, inputs=[audio, sentiment_option], outputs=[lang_str, text, sentiment_output])

        gr.HTML('''<div class="footer">
            <p>Project by <a href="https://github.com/ImATechnophile" style="text-decoration: underline;" target="_blank">Saravana Prakash</a></p>
        </div>''')

block.launch()
