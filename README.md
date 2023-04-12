# JudgerAI | Your Dream's Legal AI Assistant
Introducing **JudgerAI** - the revolutionary NLP application that predicts legal judgments with stunning accuracy! Say goodbye to the guesswork of legal decision-making and hello to unparalleled efficiency and precision. **JudgerAI** uses advanced natural language processing algorithms to analyze past cases, legal precedents, and relevant data to provide accurate predictions of future legal outcomes. With **JudgerAI**, legal professionals can make informed decisions, save time, and improve their success rates. Trust in the power of AI and let **JudgerAI** lead the way to a smarter, more efficient legal system.

Natural Language Processing (NLP) has been increasingly used in the legal field for various tasks, including predicting the outcomes of legal judgments. Legal judgment prediction involves analyzing and predicting the outcome of a legal case based on the language used in the legal documents.

**JudgerAI** can be used to analyze the language of legal cases and predict the outcome of similar cases based on patterns and trends in the language. By using **JudgerAI**, legal professionals can save time and resources by identifying relevant cases and predicting their outcome, thereby making more informed decisions.

One of the main challenges in legal judgment prediction using NLP is the complexity and variability of legal language. Legal documents often use technical terminology, jargon, and complex sentence structures that can be difficult for NLP models to analyze accurately. Additionally, legal cases can be influenced by various factors, including the specific circumstances of the case, the legal jurisdiction, and the judge's personal beliefs and biases.

Despite these challenges, NLP has shown promising results in legal judgment prediction. Researchers have used NLP techniques such as machine learning and deep learning to analyze legal language and predict the outcomes of legal cases with high accuracy. These techniques involve training NLP models on large datasets of legal cases and using them to predict the outcome of new cases based on the language used in the documents.

# Dataset
The Dataset consists of **3464** legal cases in a variety of fields, the key features of the dataset is the `first_party`, `second_party`, `winner_index`, and `facts`. here is a quick look on the dataset structure:

| column | datatype | description |
| ---    | ---      | ---         |
| ID     | int64    | Defines the case ID |
| name     | string    | Defines the case name |
| href     | string    | Defines the case hyper-reference |
| first_party     | string    | Defines the name of the first party (petitioner) of a case |
| second_party     | string    | Defines the name of the second party (respondent) of a case |
| winning_party     | string    | Defines the winning party name of a case |
| winner_index     | int64    | Defines the winning index of a case, 0 => the first party wins, 1 => the second party wins |
| facts     | string    | Contains the case facts that needed to determine who is the winner of specific case |

The input of **JudgerAI** models will be the case `facts`, and the target will be the `winner_index`.

# Modules
For organizational purposes, we divide the code base across 3 modules: `preprocessing`, `plotting`, and `utils`.
1. **preprocessing module**:
`preprocessing` module contains `Preprocessor` class which is responsible for all kind of preprocessing on the case facts such as tokenization, converting case facts to vectors using different techniques, balancing data, anonymize facts, preprocess facts, etc. **balancing - anonymization - prerprocessing** are coverd in **Experiments** section.
2. **plotting module**:
`plotting` module contains `PlottingManager` class which is responsible for all plotting & visualizations of **JudgerAI** models' performance measures including losses and accuracies curves, detailed losses and accuracies heatmaps, ROC-AUC curves, classification reports, and confusion metrics.
3. **utils module**:
`utils` module contains several useful functions that will be re-used in various models: `train_model()` function that uses k-fold cross-validation for training a specific model, `print_testing_loss_accuracy()` that summarizes testing loss and testing accuracy for each fold, `calculate_average_measure()` which is used for calculating average of the passed `measure` which can be loss, val_loss, accuracy, or val_accuracy.

# Experiments
To achieve the best results, we tried different experiments in **JudgerAI** to see each experiment's effect on the final accuracy of **JudgerAI** models, here is a list of 3 experiments that were taken into consideration:
- Data Preprocessing:
  Including removing stopwords, lowercasing all letters, stemming, removing non-alphabet characters including braces, punctuation, and digits.
- Data Anonymization:
  Replacing parties' names from the case facts with a generic `_PARTY_` tag to make sure that models are not biased towards parties' names.
- Label Class Imbalance:
  Dealing with class imbalance as a standalone preprocessing step to see if there was an impact on the final accuracy of the **JudgerAI** models or not.


Each experiment of the above 3, can be made or not, so, we ended up with 8 (2 to the power of 3) possible combinations and they were:
1. No preprocessing - No anonymization - Imbalance
2. No preprocessing - No anonymization - Balanced
3. No preprocessing - Anonymization - Imbalance
4. No preprocessing - Anonymization - Balanced
5. Preprocessing - No anonymization - Imbalance
6. Preprocessing - No anonymization - Balanced
7. Preprocessing - Anonymization - Imbalance
8. Preprocessing - Anonymization - Balanced

As a result, we will end up with 8 different results representing the effect of each expeirment on the final model's decision.

