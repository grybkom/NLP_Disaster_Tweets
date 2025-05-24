# Natural Language Processing of Disaster Tweets
Classification of Tweets as Disaster or Non-disaster Related Using Natural Language Processing (NLP)
---

![disaster_keywords](https://github.com/user-attachments/assets/713f9c1d-fe34-4711-b528-1cc4471fa78c)


## BACKGROUND

Tweets are short social networking posts broadcasted over the Twitter platform (Wikipedia contributors, 2024). In 2023 the company was purchased by Elon Musk and is now known as X (Wikipedia contributors, 2024). Early detection of disaster events can aid first responders and hasten response time. Huang et al. (2022) has proposed the use of social media posts to help rapidly identify emergency events. The aim of this work is to use natural language processing (NLP) to analyze tweets and predict if they are describing a real disaster.

## Data & Methodology

### Language
- Python
  - [Pandas](https://pandas.pydata.org/)
  - [NumPy](https://numpy.org/)
  - [Wordcloud](https://pypi.org/project/wordcloud/)
  - [Matplotlib](https://matplotlib.org/)
  - [TensorFlow](https://www.tensorflow.org/)
  - [NLTK](https://www.nltk.org/)

### Hardware
- Notebooks were created and run on Kaggle's platform using a CPU, no accelorator was used.

### Data
- The data used for this work is available at Kaggle. Addison Howard, devrishi, Phil Culliton, Yufeng Guo. (2019). Natural Language Processing with Disaster Tweets. [https://kaggle.com/competitions/nlp-getting-started](https://www.kaggle.com/competitions/nlp-getting-started/data)
- The training set consists of 7613 tweets with 4342 being labeled as non-disaster tweets and 3271 labeled as being related to a disaster. 
- The testing set consists of 3263 unlabeled tweets (for competion submission).

### Data Processing
- Tweets were converted to lowercase, and punctuation, numbers and non-alphanumeric characters were removed.
- URLs, HTML entities, and user mentions were stripped.
- Stemming was applied using the Porter Stemmer, which reduces an inflected form of a word to its root form. For example, after stemming both ‘screaming’, and ‘screamed’ would become ‘scream’.
- A Keras `Tokenizer` was used to convert words to numerical tokens, limited to the 3,500 most common words. An OOV (out-of-vocabulary) token was specified to handle unseen words.
- The tokenized tweets were converted into padded NumPy arrays to ensure consistent input lengths for modeling.
![non-disaster_keywords_comparison](https://github.com/user-attachments/assets/1aa6212d-c6bf-4d15-8650-5200dbc6b8f6)
*Figure: Most common keywords in non-disaster tweets before and after cleaning.*

### Models
**Three neural network models were constructed of increasing complexity, and their performance was compared by examining accuracy, loss, and F1 scores.**

#### 1)Simple RNN
**This baseline model serves as a starting point for comparing more complex architectures.**

**Architecture:**
- Embedding layer with vocabulary size of 3,500 and 16 dimensional embedding vectors
- SpatialDropout1D with dropout rate of 0.8 to regularize the embedding layer
- Masking layer to ignore padded values (0s)
- SimpleRNN layer with 64 units
- Dropout layer with 0.5 dropout rate
- Dense output layer with sigmoid activation for binary classification

**Training Details:**
- Loss Function: binary_crossentropy
- Optimizer: Adam with a learning rate of 0.001
- Metrics: Accuracy and F1 Score (custom implementation)
- Epochs: 24
- Batch Size: 64
- Validation Split: 20%
- Learning Rate Scheduler: ReduceLROnPlateau (to reduce LR on performance plateaus)

#### 2)Deep RNN
**This model builds on the Simple RNN by adding depth and complexity.**

- Added a second SimpleRNN layer with return_sequences=True in the first RNN to pass the entire sequence to the next layer.
- Introduced a Dense layer with 64 units and ReLU activation before the output layer, adding non-linearity and capacity.
- Applied L2 regularization to both the Dense and output layers.

All other training parameters (optimizer, loss function, metrics, epochs, batch size, learning rate schedule) remained the same as in the Simple RNN model.

#### 3) Bidirectional LSTM (BiLSTM)
**This model builds off the previous RNN architectures replacing Simple RNN layers wiht Bidirectional LSTM layers to capture context from both past and future tokens**

- Two stacked Bidirectional LSTM layers:
  - The first returns full sequences to allow the second LSTM to process temporal patterns further.
  - Both use recurrent dropout (0.3) for regularization.
- Added Batch Normalization before the output layers to stabilize and accelerate training.

All other training parameters (optimizer, loss function, metrics, epochs, batch size, learning rate schedule) remained the same as in the priveous models.

## RESULTS
### Simple RNN
![Simple RNN](https://github.com/user-attachments/assets/d604157c-5827-4c4b-8ded-2b5517cb1b44)
### Deep RNN
![RNN](https://github.com/user-attachments/assets/f3533879-9d47-4b8e-823c-7ce39b21b9cf)
### BiLSTM
![LSTM](https://github.com/user-attachments/assets/61c6187c-549c-4629-9d52-f9e53b884218)

![confusion_matrix_bidirectional](https://github.com/user-attachments/assets/026258db-9f38-411d-b925-5ddc80f71a10)

## To Do
- [ ] Develop a strategy for cleaning or handling emojis in tweets such as remove, replace with text, or encode.
- [ ] Compare lemmatization vs. stemming to determine which improves model performance during text preprocessing.
- [ ] Assess the impact of alternative learning rate schedulers, such as CosineDecayRestarts, on training stability and final performance.
- [ ] Build and evaluate a model using a pretrained transformer, such as [BERT](https://huggingface.co/docs/transformers/en/model_doc/bert) or [DistilBERT](https://arxiv.org/abs/1910.01108), for improved text understanding.
  - [ ] Visualize attention weights to interpret which words influence predictions.

## REFERENCES

Wikipedia contributors. (2024, August 3). Twitter. Wikipedia. https://en.wikipedia.org/wiki/Twitter

Huang, L., Shi, P., Zhu, H., & Chen, T. (2022). Early detection of emergency events from social media: a new text clustering approach. Natural Hazards, 111(1), 851–875. https://doi.org/10.1007/s11069-021-05081-1

## Author

Michael Grybko - GitHub username: grybkom
