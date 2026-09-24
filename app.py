
import streamlit as st

from nmt_translator import translate

st.set_page_config(page_title="English-Amharic Translator", page_icon="🌍")

st.title("English to Amharic Translator")
st.write("Enter an English sentence and translate it with the trained NMT model.")

text = st.text_area(
    "English text",
    placeholder="I am going to the university.",
    height=140,
)

if st.button("Translate", type="primary"):
    if not text.strip():
        st.warning("Please enter some English text.")
    else:
        with st.spinner("Translating..."):
            try:
                result = translate(text)
            except Exception as error:
                st.error(f"Translation failed: {error}")
            else:
                st.subheader("Amharic translation")
                st.success(result or "No translation was generated.")
