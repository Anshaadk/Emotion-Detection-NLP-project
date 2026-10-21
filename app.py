import pickle
import string
from pathlib import Path

import streamlit as st

MODEL_PATH = Path(__file__).parent / "emotion_model.pkl"

EMOTION_EMOJIS = {
    "joy": "😊",
    "happiness": "😊",
    "happy": "😊",
    "sadness": "😢",
    "sad": "😢",
    "anger": "😠",
    "angry": "😠",
    "fear": "😨",
    "love": "❤️",
    "surprise": "😲",
    "disgust": "🤢",
    "neutral": "😐",
    "shame": "😳",
    "guilt": "😔",
    "frustration": "😤",
    "sarcasm": "🙃",
}

EXAMPLES = {
    "Joy 😊": "I am so happy today, everything is going well!",
    "Sadness 😢": "I feel lonely and I miss my friends.",
    "Anger 😠": "I am really angry because nobody listened to me.",
    "Fear 😨": "I am scared about what might happen tomorrow.",
    "Love ❤️": "I care about you so much and I am grateful you are here.",
    "Surprise 😲": "Wow, I never expected this amazing news!",
    "Sarcasm 🙃": "Fantastic, another meeting that could have been an email.",
}

st.set_page_config(
    page_title="Emotion Detector",
    page_icon="💬",
    layout="centered",
)

st.title("💬 Emotion Detector")
st.write("Type a sentence or choose an example to see the emotions your model predicts.")

if not MODEL_PATH.exists():
    st.error("I couldn't find emotion_model.pkl.")
    st.write("Put emotion_model.pkl in the same folder as app.py.")
    st.stop()

@st.cache_resource
def load_artifacts(path):
    with open(path, "rb") as file:
        return pickle.load(file)

artifacts = load_artifacts(str(MODEL_PATH))
model = artifacts["model"]
vectorizer = artifacts["vectorizer"]
id_to_emotion = artifacts["id_to_emotion"]
stop_words = set(artifacts.get("stop_words", set()))

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.translate(str.maketrans("", "", string.digits))
    text = "".join(char for char in text if char.isascii())
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

def emotion_emoji(emotion):
    return EMOTION_EMOJIS.get(str(emotion).lower(), "💭")

example_choice = st.selectbox(
    "Try an example",
    ["Choose an example..."] + list(EXAMPLES.keys()),
)

if st.button("Load example"):
    if example_choice != "Choose an example...":
        st.session_state["user_text"] = EXAMPLES[example_choice]

user_text = st.text_area(
    "Your sentence",
    key="user_text",
    placeholder="Example: I feel really happy today!",
    height=130,
)

if st.button("Predict emotions", type="primary", use_container_width=True):
    if not user_text.strip():
        st.warning("Please enter a sentence first.")
    else:
        cleaned = clean_text(user_text)
        if not cleaned:
            st.warning("No usable words remained after cleaning. Try another sentence.")
        else:
            features = vectorizer.transform([cleaned])
            predicted_id = int(model.predict(features)[0])
            predicted_emotion = id_to_emotion.get(
                predicted_id, str(predicted_id)
            )

            st.subheader("Predicted emotion")
            st.success(
                f"{emotion_emoji(predicted_emotion)} "
                f"{str(predicted_emotion).title()}"
            )

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(features)[0]
                class_ids = model.classes_

                scores = []
                for class_id, probability in zip(class_ids, probabilities):
                    label = id_to_emotion.get(int(class_id), str(class_id))
                    scores.append((label, float(probability)))

                scores.sort(key=lambda item: item[1], reverse=True)

                st.subheader("All emotion scores")
                st.caption(
                    "These are the model's scores, not guaranteed probabilities "
                    "of the true emotion. Sarcasm and context can confuse the model."
                )

                for label, probability in scores:
                    left, right = st.columns([3, 1])
                    with left:
                        st.write(f"{emotion_emoji(label)} **{str(label).title()}**")
                    with right:
                        st.write(f"**{probability:.1%}**")
                    st.progress(max(0.0, min(1.0, probability)))
            else:
                st.info("This saved model does not provide probability scores.")

st.divider()
st.subheader("Example sentences")
for label, example in EXAMPLES.items():
    st.markdown(f"**{label}**")
    st.write(f"“{example}”")

st.caption(
    "Learning project: the model predicts only the emotion classes it was trained on. "
    "Sarcasm examples may still be misclassified unless the training data covers sarcasm."
)