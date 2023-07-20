"""
#* make sure to run the application using this commmand:
>>> streamlit run main.py
"""

# global
import streamlit as st
from lime.lime_text import LimeTextExplainer
from nltk.corpus import stopwords

# local
from deployment_utils import DataPreparator, Predictor, generate_random_sample, generate_highlighted_words


# instantiate `DataPreparator` & `Predictor` objects
data_preparator = DataPreparator()
predictor = Predictor()
eng_stop_words = stopwords.words("english")

# for custom CSS styling
with open("F:/Graduation Project/Project/src/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# application header
st.header("AI in the Courtroom | An Intelligent Judicial System")

get_random_case_button = st.button(label="Get Random Sample")

# input form
with st.form(key="inputs_form", clear_on_submit=False):
    option = st.selectbox(
        "Select Model - اختر النموذج",
        ("TF-IDF", "1D Convolutional", "GloVe", "BERT", "Doc2Vec",
         "LSTM", "FastText", "Ensemble (Doc2Vec + TF-IDF)")
    )

    # if `Get Random Sample` btn is pressed
    if get_random_case_button:
        random_petitioner, random_respondent, random_facts, random_label = generate_random_sample()
        st.session_state["petitioner"] = random_petitioner.strip()
        st.session_state["respondent"] = random_respondent.strip()
        st.session_state["facts"] = random_facts.strip()
        st.success(f"Original label: {random_label}")

    col1, col2 = st.columns(2)

    with col1:
        petitioner = st.text_input(
            label="Petitioner - الملتمس", key="petitioner")

    with col2:
        respondent = st.text_input(
            label="Respondent - المدعى عليه", key="respondent")

    facts = st.text_area(label="Case Facts - ادلة القضية",
                         height=300, key="facts")

    # remove stopwords to not highlight it
    facts = " ".join([word for word in facts.split()
                     if word not in eng_stop_words])

    # create `LimeTextExplainer` for models interpretation
    class_names = [petitioner, respondent]
    explainer = LimeTextExplainer(class_names=class_names)

    submitted = st.form_submit_button(label="Submit")

    if submitted:
        if petitioner and respondent and facts:
            with st.spinner("Analyzing Case Facts ..."):
                # get predcitions
                if option == "Doc2Vec":
                    predictions = predictor.predict_doc2vec(facts)
                    output = explainer.explain_instance(
                        facts, predictor.predict_doc2vec)
                    important_words = output.as_list()

                elif option == "TF-IDF":
                    anonymized_facts = data_preparator._anonymize_facts(
                        petitioner, respondent, facts)
                    predictions = predictor.predict_tf_idf(anonymized_facts)
                    output = explainer.explain_instance(
                        anonymized_facts, predictor.predict_tf_idf)
                    important_words = output.as_list()

                elif option == "1D Convolutional":
                    predictions = predictor.predict_cnn(facts)
                    output = explainer.explain_instance(
                        facts, predictor.predict_cnn)
                    important_words = output.as_list()

                elif option == "GloVe":
                    predictions = predictor.predict_glove(facts)
                    output = explainer.explain_instance(
                        facts, predictor.predict_glove)
                    important_words = output.as_list()

                elif option == "LSTM":
                    predictions = predictor.predict_lstm(facts)
                    output = explainer.explain_instance(
                        facts, predictor.predict_lstm)
                    important_words = output.as_list()

                elif option == "BERT":
                    predictions = predictor.predict_bert(facts)

                elif option == "FastText":
                    predictions = predictor.predict_fasttext(facts)

                elif option == "Ensemble (Doc2Vec + TF-IDF)":
                    doc2vec_predictions = predictor.predict_doc2vec(facts)
                    tf_idf_predictions = predictor.predict_tf_idf(facts)

                    predictions = (doc2vec_predictions +
                                   tf_idf_predictions) / 2

                    doc2vec_output = explainer.explain_instance(
                        facts, predictor.predict_doc2vec)
                    doc2vec_important_words = doc2vec_output.as_list()

                    tf_idf_output = explainer.explain_instance(
                        facts, predictor.predict_tf_idf)
                    tf_idf_important_words = tf_idf_output.as_list()

                    important_words = doc2vec_important_words + tf_idf_important_words

            # displaying predictions
            col1, col2 = st.columns(2)

            with col1:
                st.write("Percentage of petitioner winning:")
                st.warning(f"{predictions[0, 0] * 100:.3f}%")

            with col2:
                st.write("Percentage of respondent winning:")
                st.info(f"{predictions[0, 1] * 100:.3f}%")

            st.write("Winning party:")
            if predictions[0, 0] > predictions[0, 1]:
                st.success(petitioner)
            else:
                st.success(respondent)

            # displaying highlighted words
            st.write(
                '<p class="bold-text"> Top words for model\'s decision: </p>', unsafe_allow_html=True)

            if option not in ["BERT", "FastText"]:
                petitioner_words = [word for word,
                                    score in important_words if score < 0]
                respondent_words = [word for word,
                                    score in important_words if score > 0]

                for name in petitioner.split(" "):
                    if name in petitioner_words:
                        petitioner_words.remove(name)
                    elif name in respondent_words:
                        respondent_words.remove(name)

                for name in respondent.split(" "):
                    if name in petitioner_words:
                        petitioner_words.remove(name)
                    elif name in respondent_words:
                        respondent_words.remove(name)

                rendered_text = generate_highlighted_words(
                    facts, petitioner_words, respondent_words)

                st.write(rendered_text, unsafe_allow_html=True)

            else:
                st.warning(
                    "Sadly, this feature is not supported in BERT & FastText :(")

        else:
            st.error("Please, fill in all fields!")
