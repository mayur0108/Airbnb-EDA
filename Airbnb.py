#!/usr/bin/env python
# coding: utf-8

# In[ ]:


nltk.download('punkt') 
nltk.download('wordnet') 


# In[1]:


nltk.download('averaged_perceptron_tagger')
#pip install gensim
pip inst
all tensorflow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install cuml


# In[1]:


import nltk
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import sys
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk import pos_tag






# # code Initialization

# In[2]:


# Hide deprecated warnings of sklearn package
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


# In[ ]:





# In[3]:


true_csv_path = 'true.csv'  
fake_csv_path = 'fake.csv'  

true_df = pd.read_csv(true_csv_path)
fake_df = pd.read_csv(fake_csv_path)

true_df['label'] = '0'
fake_df['label'] = '1'

true_df.to_csv(true_csv_path, index=False)
fake_df.to_csv(fake_csv_path, index=False)


# In[4]:


true_df.head()


# In[5]:


fake_df.head()


# In[6]:


# Merge DataFrames
merged_df = pd.concat([true_df, fake_df], ignore_index=True)


merged_df = merged_df.sample(frac=1).reset_index(drop=True) #randomized 

merged_csv_path = 'merged.csv'  
merged_df.to_csv(merged_csv_path, index=False)

print(f'Merged file saved to: {merged_csv_path}')


# In[7]:


merge_csv_path = 'merged.csv'
df = pd.read_csv(merge_csv_path)


# In[8]:


df.tail()


# # Data cleaning 
# 
# 1.1.	Removing HTML tags or unwanted characters
#     1.1.1.	Eliminating special characters, punctuation
#     1.1.2.	remove common English stopwords.
# 1.2.	Converting text to lowercase
# 1.3.	finding rows with missing values
#     1.3.1.	Filling missing values with a placeholder
#     1.3.2.	Imputing missing values
#     1.3.3.	Removing rows with unnecessary missing values
# 1.4.	Text Length (Characters)
#     1.4.1.	Word Count:
# 

# In[9]:


# Function to clean text
def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert text to lowercase
    text = text.lower().strip()
    return text

df['cleaned_title'] = df['title'].apply(clean_text)
df['cleaned_text'] = df['text'].apply(clean_text)




# In[10]:


df.tail()


# In[11]:


# Load stop words
stop_words = set(stopwords.words('english'))

# Function to remove stop words
def remove_stop_words(text):
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return ' '.join(filtered_text)


df['cleaned_title_no_stopwords'] = df['cleaned_title'].apply(remove_stop_words)
df['cleaned_text_no_stopwords'] = df['cleaned_text'].apply(remove_stop_words)


# In[12]:


df.drop(['cleaned_text', 'cleaned_title',], axis=1, inplace=True)



# In[13]:


df.tail()


# # Steemming

# In[14]:


#stemming
# Initialize the Porter Stemmer
stemmer = PorterStemmer()

#stemming to a text
def stem_text(text):
    word_tokens = word_tokenize(text)
    stemmed_text = [stemmer.stem(word) for word in word_tokens]
    return ' '.join(stemmed_text)

#progress visualization
def stem_with_progress(data, column_name):
    stemmed_data = []
    total = len(data)
    print("Starting stemming process...")
    for i, text in enumerate(data[column_name], 1):
        stemmed_data.append(stem_text(text))
        if i % 100 == 0 or i == total:  
            sys.stdout.write('\rProgress: {0:.2f}%'.format(100 * i/total))
            sys.stdout.flush()
    print("\nStemming process completed.")
    return stemmed_data


df['stemmed_title'] = stem_with_progress(df, 'cleaned_title_no_stopwords')
df['stemmed_text'] = stem_with_progress(df, 'cleaned_text_no_stopwords')


# In[15]:


df.head()


# # Lemmetization

# In[16]:


#lemmetization
# Initialize the WordNet Lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to convert NLTK's part-of-speech tags to WordNet's part-of-speech tags
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun if unknown

# apply lemmatization to a text
def lemmatize_text(text):
    word_tokens = word_tokenize(text)
    pos_tagged_tokens = pos_tag(word_tokens)
    lemmatized_text = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tagged_tokens]
    return ' '.join(lemmatized_text)

# progress visualization
def lemmatize_with_progress(data, column_name):
    lemmatized_data = []
    total = len(data)
    print("Starting lemmatization process...")
    for i, text in enumerate(data[column_name], 1):
        lemmatized_data.append(lemmatize_text(text))
        if i % 100 == 0 or i == total:  # Update progress every 100 items or at the end
            sys.stdout.write('\rProgress: {0:.2f}%'.format(100 * i/total))
            sys.stdout.flush()
    print("\nLemmatization process \completed.")
    return lemmatized_data


df['lemmatized_title'] = lemmatize_with_progress(df, 'stemmed_title')
df['lemmatized_text'] = lemmatize_with_progress(df, 'stemmed_text')





# In[17]:


df.head()


# In[18]:


# Check for missing values 
missing_values = df.isnull().sum()


print(missing_values)


# In[19]:


df.drop(['cleaned_title_no_stopwords', 'cleaned_text_no_stopwords','stemmed_title','stemmed_text'], axis=1, inplace=True)


# In[20]:


df.head()


# In[ ]:





# # date conversion 

# In[21]:


pip install python-dateutil


# In[22]:


from dateutil import parser


def safe_parse_date(x):
    try:
        
        if isinstance(x, str) and not x.startswith('http'):
            return parser.parse(x).strftime('%m/%d/%Y')
    except ValueError:
        pass
    return None  

df['date'] = df['date'].apply(lambda x: safe_parse_date(x))


print(df['date'].head())


# In[23]:


df.drop(['text', 'title',], axis=1, inplace=True)
df.head()


# In[31]:


df['subject'].value_counts()


# In[32]:


# Remove rows where the 'date' column has NaN values
df = df.dropna(subset=['date'])

# Verify the removal by checking for NaN values again
print(df['date'].isnull().sum())


# In[33]:


df.head()


# # sentiment ANALYSIS

# In[34]:


import pandas as pd
from textblob import TextBlob

df['sentiment'] = df['lemmatized_text'].apply(lambda text: TextBlob(text).sentiment.polarity)


# In[35]:


for index, row in df.iterrows():
    sentiment_score = row['sentiment']
    sentiment_score2 = row['label']
    print(f"Document {index} has a sentiment polarity of {sentiment_score} and the sentiment score is {sentiment_score2}")

    if sentiment_score < 0:
        print("The sentiment is negative.")
    elif sentiment_score > 0:
        print("The sentiment is positive.")
    else:
        print("The sentiment is neutral.")


# In[36]:


import pandas as pd


df['sentiment'] = df['sentiment'].astype(str)


# In[37]:


df.dtypes


# # Vectorization

# In[ ]:





# In[38]:


from gensim.models.doc2vec import TaggedDocument
#import pandas as pd

# Assuming 'df' is your DataFrame and 'cleaned_text_no_stopwords' is the column with preprocessed text
documents = [TaggedDocument(words=text.split(), tags=[i]) for i, text in enumerate(df['lemmatized_title'])]


# In[39]:


import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


lemmatized_docs = (df['subject']+" " +df['lemmatized_title'] + " " + df['lemmatized_text']+" "+ df['sentiment']).apply(lambda x: x.split())
print(lemmatized_docs)

tagged_title_data = [TaggedDocument(words=words, tags=[str(i)]) for i, words in enumerate(lemmatized_docs)]

max_epochs = 100
vec_size = 300
alpha = 0.025

model = Doc2Vec(vector_size=vec_size,
                alpha=alpha,
                min_alpha=0.00025,  # Gradual decay to the minimum alpha
                min_count=5,
                window=10,
                dm=1,
                )

model.build_vocab(tagged_title_data)
print('Training Doc2Vec Model')

for epoch in range(max_epochs):
    print(f'Training epoch {epoch + 1}/{max_epochs}')
    model.train(tagged_title_data,
                total_examples=model.corpus_count,
                epochs=1)  # Train for one epoch at a time
    model.alpha -= (alpha - model.min_alpha) / max_epochs  # Decrease the learning rate
    model.min_alpha = model.alpha  # Fix the learning rate, no decay

# Save the model
model.save("d2v_all_all21.model")
print("Model Saved")


# In[40]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

model = Doc2Vec.load("d2v_all_all21.model")


X = [model.dv[str(i)] for i in range(len(df))]
y = df['label'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)

report = classification_report(y_test, y_pred, output_dict=True)
f1_score = report['macro avg']['f1-score']  # Use 'weighted' if you want to consider label imbalance

print(f"F1 Score: {f1_score}")
print(classification_report(y_test, y_pred))
print(y.shape)


# In[41]:


import numpy as np
from gensim.models.doc2vec import Doc2Vec

# Load the Doc2Vec model
model = Doc2Vec.load("d2v_all_all21.model")



# Function to infer the mean word vector for words in a document
def infer_mean_vector(doc_words):

    if len(doc_words) == 0:
        return np.zeros(model.vector_size)
   
    return np.mean([model.wv[word] for word in doc_words if word in model.wv.key_to_index], axis=0)

# Combine document and word vectors
combined_vectors = []
for i, doc_words in enumerate(documents):
   
    if isinstance(doc_words[0], list):  
        doc_words = [word for sublist in doc_words for word in sublist] 


    doc_vector = model.dv[str(i)]
    mean_word_vector = infer_mean_vector(doc_words)


    combined_vector = np.concatenate((doc_vector, mean_word_vector))
    combined_vectors.append(combined_vector)



np_combined_vectors = np.array(combined_vectors)

print('Shape of the combined data:', np_combined_vectors.shape)


# # Model Building

# In[45]:


# Extract labels from the DataFrame
labels = df['label'].values 

# Now, proceed with splitting the data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(np_combined_vectors, labels, test_size=0.2, random_state=42)




# In[36]:


import numpy as np

X_np = np.array(X)
labels_np = np.array(labels)


# ## SVM ON DOC2VEC

# In[37]:


from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from tqdm import tqdm

X_np = np.array(X)
labels_np = np.array(labels)

# Initialize KFold cross-validation
kf = KFold(n_splits=5)

accuracies = []

for train_index, test_index in tqdm(kf.split(X_np), total=kf.get_n_splits(), desc='KFold Progress'):
   
    X_train, X_test = X_np[train_index], X_np[test_index]
    y_train, y_test = labels_np[train_index], labels_np[test_index]
    
    # Initialize and train the SVM classifier on the training set
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train, y_train)
    
    # Predict on the testing set
    y_pred = svm_classifier.predict(X_test)
    
    # Calculate and print the accuracy and classification report for the current fold
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print(f"Fold Accuracy: {accuracy}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# Print the average accuracy over all folds
print(f"Average KFold Accuracy: {np.mean(accuracies)}")


# In[ ]:





# In[ ]:





# ## Simple  binary classification

# In[39]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm


labels = df['label'].values.astype(float)

scaler = StandardScaler()
np_combined_vectors_scaled = scaler.fit_transform(np_combined_vectors)

X_train, X_test, y_train, y_test = train_test_split(np_combined_vectors_scaled, labels, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

class BinaryClassifier(nn.Module):
    def __init__(self, n_features):
        super(BinaryClassifier, self).__init__()
        self.linear = nn.Linear(n_features, 1)  # Output is logits for BCEWithLogitsLoss
    
    def forward(self, x):
        return self.linear(x)

n_features = X_train.shape[1]
model = BinaryClassifier(n_features)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor.view(-1, 1))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

model.train()
n_epochs = 20  
for epoch in tqdm(range(n_epochs), desc='Epochs'):
    running_loss = 0.0
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {running_loss / len(train_loader)}")

    model.eval()
    with torch.no_grad():
        outputs = model(X_train_tensor)
        predicted_probs = torch.sigmoid(outputs).view(-1)
        predicted_labels = torch.round(predicted_probs)  # Convert probabilities to binary labels
        train_accuracy = accuracy_score(y_train_tensor.numpy(), predicted_labels.numpy())
        print(f"Training Accuracy: {train_accuracy}")

    model.train()  

torch.save(model.state_dict(), "binary_classifier_model.pth")
print("Model saved to binary_classifier_model.pth")

model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    predicted_probs = torch.sigmoid(outputs).view(-1)
    predicted_labels = torch.round(predicted_probs)  # Convert probabilities to binary labels
    test_accuracy = accuracy_score(y_test_tensor.numpy(), predicted_labels.numpy())
    print(f"Test Accuracy: {test_accuracy}")
    print("Final Test Classification Report:")
    print(classification_report(y_test_tensor.numpy(), predicted_labels.numpy(), digits=4))



# In[ ]:





# In[ ]:





# ## Naive bayes

# In[65]:


# naive bayes using sckit learn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd

labels = df['label'].values 


np_combined_vectors += np.abs(np_combined_vectors.min())

X_train, X_test, y_train, y_test = train_test_split(np_combined_vectors, labels, test_size=0.2, random_state=42)

nb_classifier = MultinomialNB()

nb_classifier.fit(X_train, y_train)

y_pred = nb_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred))



kappa = cohen_kappa_score(y_test, y_pred)
print(f"Cohen's Kappa Score: {kappa}")



# # Support vector machine

# In[43]:


#SVM using ski-it learn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score, classification_report, matthews_corrcoef


labels = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(np_combined_vectors, labels, test_size=0.2, random_state=42)

svm_classifier = SVC(kernel='linear', max_iter=10000, tol=1e-3)


svm_classifier.fit(X_train, y_train)

y_pred = svm_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred))


mcc = matthews_corrcoef(y_test, y_pred)
print(f"Matthews Correlation Coefficient: {mcc}")

kappa = cohen_kappa_score(y_test, y_pred)
print(f"Cohen's Kappa Score: {kappa}")


# In[ ]:





# In[53]:


pip install joblib


# In[46]:


import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, matthews_corrcoef
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib  


scaler = StandardScaler()
np_combined_vectors_scaled = scaler.fit_transform(np_combined_vectors)


#joblib.dump(scaler, 'scaler.joblib')


k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

accuracies = []
kappa_scores = []
mcc_scores = []

best_accuracy = 0  

for fold, (train_index, test_index) in enumerate(kf.split(np_combined_vectors_scaled), 1):
    X_train, X_test = np_combined_vectors_scaled[train_index], np_combined_vectors_scaled[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    accuracies.append(accuracy)
    kappa_scores.append(kappa)
    mcc_scores.append(mcc)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = svm_classifier  # Update the best model
    
    print(f"Fold {fold}")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy}")
    print(f"Cohen's Kappa Score: {kappa}")
    print(f"Matthews Correlation Coefficient: {mcc}")
    print("-" * 30)

model_filename = 'best_svm_model.joblib'
joblib.dump(best_model, model_filename)
print(f"Best model saved to {model_filename}")

print(f"Average Accuracy across all folds: {np.mean(accuracies)}")
print(f"Average Cohen's Kappa Score across all folds: {np.mean(kappa_scores)}")
print(f"Average Matthews Correlation Coefficient across all folds: {np.mean(mcc_scores)}")


# In[34]:


import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, matthews_corrcoef
from sklearn.model_selection import KFold, train_test_split
import pandas as pd
import joblib  


k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

accuracies = []
kappa_scores = []
mcc_scores = []

best_accuracy = 0  

for fold, (train_index, test_index) in enumerate(kf.split(np_combined_vectors), 1):
    X_train, X_test = np_combined_vectors[train_index], np_combined_vectors[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    accuracies.append(accuracy)
    kappa_scores.append(kappa)
    mcc_scores.append(mcc)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = svm_classifier  # Update the best model
    
    print(f"Fold {fold}")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy}")
    print(f"Cohen's Kappa Score: {kappa}")
    print(f"Matthews Correlation Coefficient: {mcc}")
    print("-" * 30)

model_filename = 'best_svm_model.joblib'
joblib.dump(best_model, model_filename)
print(f"Best model saved to {model_filename}")

print(f"Average Accuracy across all folds: {np.mean(accuracies)}")
print(f"Average Cohen's Kappa Score across all folds: {np.mean(kappa_scores)}")
print(f"Average Matthews Correlation Coefficient across all folds: {np.mean(mcc_scores)}")


# ## Decision Tree classifier 

# In[44]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, matthews_corrcoef
import numpy as np
import pandas as pd



labels = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(np_combined_vectors, labels, test_size=0.2, random_state=42)

dt_classifier = DecisionTreeClassifier(random_state=42)

dt_classifier.fit(X_train, y_train)

y_pred = dt_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

kappa = cohen_kappa_score(y_test, y_pred)
print(f"Cohen's Kappa Score: {kappa}")

mcc = matthews_corrcoef(y_test, y_pred)
print(f"Matthews Correlation Coefficient: {mcc}")


# In[46]:


# decision tree using kfold from skilearn
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, matthews_corrcoef
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import pandas as pd


scaler = StandardScaler()
np_combined_vectors_scaled = scaler.fit_transform(np_combined_vectors)

k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=2)

accuracies = []
kappa_scores = []
mcc_scores = []

for fold, (train_index, test_index) in enumerate(kf.split(np_combined_vectors_scaled), 1):
    X_train, X_test = np_combined_vectors_scaled[train_index], np_combined_vectors_scaled[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    
    dt_classifier = DecisionTreeClassifier(random_state=2)
    
    dt_classifier.fit(X_train, y_train)
    
    y_pred = dt_classifier.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    accuracies.append(accuracy)
    kappa_scores.append(kappa)
    mcc_scores.append(mcc)
    
    print(f"Fold {fold}")
    print(f"Accuracy: {accuracy}")
    print(f"Cohen's Kappa Score: {kappa}")
    print(f"Matthews Correlation Coefficient: {mcc}")
    print("-" * 30)

print(f"Average Accuracy across all folds: {np.mean(accuracies)}")
print(f"Average Cohen's Kappa Score across all folds: {np.mean(kappa_scores)}")
print(f"Average Matthews Correlation Coefficient across all folds: {np.mean(mcc_scores)}")


# # k nearest neighbour

# In[47]:


#knn without fold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
import numpy as np


scaler = StandardScaler()
np_combined_vectors_scaled = scaler.fit_transform(np_combined_vectors)


X_train, X_test, y_train, y_test = train_test_split(np_combined_vectors_scaled, labels, test_size=0.20, random_state=1000)


knn_classifier = KNeighborsClassifier(n_neighbors=2)

knn_classifier.fit(X_train, y_train)

y_pred = knn_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"KNN Test Accuracy: {accuracy}")
print("KNN Classification Report:")
print(classification_report(y_test, y_pred))


kappa = cohen_kappa_score(y_test, y_pred)
print(f"Cohen's Kappa Score: {kappa}")
mcc = matthews_corrcoef(y_test, y_pred)
print(f"Matthews Correlation Coefficient: {mcc}")


# In[48]:


#knn with folds 
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
import numpy as np



scaler = StandardScaler()
np_combined_vectors_scaled = scaler.fit_transform(np_combined_vectors)


k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)


fold_accuracies = []
fold_kappa_scores = []
fold_mcc_scores = []

for fold, (train_index, test_index) in enumerate(kf.split(np_combined_vectors_scaled), 1):

    X_train, X_test = np_combined_vectors_scaled[train_index], np_combined_vectors_scaled[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    
    knn_classifier = KNeighborsClassifier(n_neighbors=2)
    
    knn_classifier.fit(X_train, y_train)
    
    y_pred = knn_classifier.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    
    fold_accuracies.append(accuracy)
    fold_kappa_scores.append(kappa)
    fold_mcc_scores.append(mcc)
    
    
    print(f"Fold {fold}")
    print(f"Accuracy: {accuracy}")
    print(f"Cohen's Kappa Score: {kappa}")
    print(f"Matthews Correlation Coefficient: {mcc}")
    print("-" * 30)


print(f"Average Accuracy across all folds: {np.mean(fold_accuracies)}")
print(f"Average Cohen's Kappa Score across all folds: {np.mean(fold_kappa_scores)}")
print(f"Average Matthews Correlation Coefficient across all folds: {np.mean(fold_mcc_scores)}")


# In[ ]:





# In[34]:


#knn with param_grid and model saving
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import joblib  



X_train, X_test, y_train, y_test = train_test_split(np_combined_vectors, labels, test_size=0.2, random_state=42)

param_grid = {
    'n_neighbors': [2, 3, 5, 7, 10],
    'metric': ['euclidean', 'manhattan'],
    'weights': ['uniform', 'distance'],
    'algorithm': ['ball_tree', 'kd_tree', 'brute']
}


knn = KNeighborsClassifier()

grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)

grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_}")


best_knn = grid_search.best_estimator_
joblib.dump(best_knn, 'best_knn_model.joblib')


y_pred = best_knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"KNN Test Accuracy (Optimized): {accuracy}")
print("KNN Classification Report (Optimized):")
print(classification_report(y_test, y_pred))


# # Logistic regression

# In[40]:


# logistic regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import joblib



X_train, X_test, y_train, y_test = train_test_split(np_combined_vectors, labels, test_size=0.2, random_state=42)

logistic_model = LogisticRegression(max_iter=1000, random_state=42)

logistic_model.fit(X_train, y_train)

y_pred = logistic_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Test Accuracy: {accuracy}")
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred))


model_filename = 'logistic_regression_model.joblib'
joblib.dump(logistic_model, model_filename)
print(f"Model saved to {model_filename}")


# In[41]:


from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, matthews_corrcoef
import numpy as np
import joblib



k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)


fold_accuracies = []
fold_kappa_scores = []
fold_mcc_scores = []


for train_index, test_index in kf.split(np_combined_vectors):
    X_train, X_test = np_combined_vectors[train_index], np_combined_vectors[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    
    logistic_model = LogisticRegression(max_iter=1000, random_state=42)
    
    logistic_model.fit(X_train, y_train)
    
    y_pred = logistic_model.predict(X_test)
    
    fold_accuracies.append(accuracy_score(y_test, y_pred))
    fold_kappa_scores.append(cohen_kappa_score(y_test, y_pred))
    fold_mcc_scores.append(matthews_corrcoef(y_test, y_pred))


print(f"Average Logistic Regression Test Accuracy: {np.mean(fold_accuracies)}")
print(f"Average Cohen's Kappa Score across all folds: {np.mean(fold_kappa_scores)}")
print(f"Average Matthews Correlation Coefficient across all folds: {np.mean(fold_mcc_scores)}")



# # Gradient Boosting

# In[ ]:


from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import make_scorer, accuracy_score, cohen_kappa_score, matthews_corrcoef
import numpy as np
import joblib



gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)


k_folds = 5
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

scorers = {
    'accuracy_score': make_scorer(accuracy_score),
    'cohen_kappa_score': make_scorer(cohen_kappa_score),
    'matthews_corrcoef': make_scorer(matthews_corrcoef)
}


cv_results = {}
for scorer_name, scorer in scorers.items():
    cv_score = cross_val_score(gbm, np_combined_vectors, labels, cv=skf, scoring=scorer, n_jobs=-1)
    cv_results[scorer_name] = cv_score
    print(f"Average {scorer_name}: {np.mean(cv_score)}")


gbm.fit(np_combined_vectors, labels)
joblib.dump(gbm, 'gbm_model.joblib')


# # model prediction on external data

# In[38]:


import numpy as np
from gensim.models.doc2vec import Doc2Vec
import joblib
from textblob import TextBlob
import re

doc2vec_model = Doc2Vec.load("d2v_all_all21.model")
classifier_model = joblib.load("best_svm_model.joblib")

def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.strip()

def calculate_sentiment(text):
    
    sentiment = TextBlob(text).sentiment.polarity
    return str(sentiment)  # Convert sentiment score to string

def infer_mean_vector(doc_words, model):
    if len(doc_words) == 0:
        return np.zeros(model.vector_size)
    return np.mean([model.wv[word] for word in doc_words if word in model.wv.key_to_index], axis=0)

def vectorize_and_combine(text, model):
    sentiment_str = calculate_sentiment(text['text'])
    combined_text = f"{text['subject']} {text['title']} {text['text']} {sentiment_str}"
    preprocessed_text = preprocess_text(combined_text)
    words = preprocessed_text.split()
    doc_vector = model.infer_vector(words)
    mean_word_vector = infer_mean_vector(words, model)
    combined_vector = np.concatenate((doc_vector, mean_word_vector))
    return combined_vector

test_data = [
    {'subject': "worldnews", 'title': "China, South Korea agree to mend ties after THAAD standoff", 'text': "SEOUL/BEIJING (Reuters) - Seoul and Beijing on Tuesday agreed to move beyond a year-long stand-off over the deployment of a U.S. anti-missile system in South Korea, a dispute that has been devastating to South Korean businesses that rely on Chinese consumers. The unexpected detente comes just days before U.S. President Donald Trump begins a trip to Asia, where the North Korean nuclear crisis will take center stage, and helped propel South Korean stocks to a record high. The installation of the U.S. Terminal High Altitude Area Defense (THAAD) system had angered China, with South Korea s tourism, cosmetics and entertainment industries bearing the brunt of a Chinese backlash, although Beijing has never specifically linked that to the THAAD deployment. Beijing worries the THAAD system s powerful radar can penetrate into Chinese territory.  Both sides shared the view that the strengthening of exchange and cooperation between Korea and China serves their common interests and agreed to expeditiously bring exchange and cooperation in all areas back on a normal development track,  South Korea s foreign ministry said in a statement. Before the THAAD dispute, bilateral relations flourished,  despite Beijing s historic alliance with North Korea and Seoul s close ties with Washington, which includes hosting 28,500 U.S. troops. China is South Korea s biggest trading partner.  At this critical moment all stakeholders should be working together to address the North Korea nuclear challenge instead of creating problems for others,  said Wang Dong, associate professor of international studies at China s Peking University.  This sends a very positive signal that Beijing and Seoul are determined to improve their relations.  As part of the agreement, South Korean President Moon Jae-in will meet Chinese President Xi Jinping on the sidelines of the summit of Asia-Pacific Economic Cooperation (APEC) countries in Vietnam on Nov. 10-11. South Korea recognized China s concerns over THAAD and made it clear the deployment was not aimed at any third country and did not harm China s strategic security interests, China s foreign ministry said. China reiterated its opposition to the deployment of THAAD, but noted South Korea s position and hoped South Korea could appropriately handle the issue, it added.  China s position on the THAAD issue is clear, consistent and has not changed,  Chinese Foreign Ministry spokeswoman Hua Chunying told a daily briefing in Beijing. The thaw is a big relief for South Korean tourism and retail firms as well as K-pop stars and makers of films and soap operas, which had found themselves unofficially unwelcome in China over the past year. In South Korea, a halving of inbound Chinese tourists in the first nine months of the year cost the economy $6.5 billion in lost revenue based on the average spending of Chinese visitors in 2016, data from the Korea Tourism Organization shows. The spat knocked about 0.4 percentage points off this year s expected economic growth, according to the Bank of Korea, which now forecasts an expansion of 3 percent. The sprawling Lotte Group, which provided the land where the THAAD battery was installed and is a major operator of hotels and duty free stores, has been hardest hit. It faces a costly overhaul and is expected to sell its Chinese hypermarket stores for a fraction of what it invested. A spokesman for holding company Lotte Corp expressed hope that South Korean firms  activity in China would improve following the announcement. An official at Seoul s presidential Blue House, who declined to be named given the sensitivity of the matter, said improvements for South Korean companies would come slowly. Shares in South Korean tourism and retail companies rallied nonetheless, with Asiana Airlines gaining 3.6 percent and Lotte Shopping up 7.14 percent. The benchmark Kospi index hit a record for a third straight day, gaining 0.9 percent. China has grown increasingly angry with North Korea s ongoing pursuit of nuclear weapons and ballistic missiles in defiance of United Nations sanctions, even as it chafes at U.S. pressure to rein in its isolated ally. The recent deterioration in ties between China and North Korea may have contributed to Tuesday s agreement, the Blue House official said. Pyongyang has undertaken an unprecedented missile testing program in recent months, as well as its biggest nuclear test yet in early September, as it seeks to develop a powerful nuclear weapon capable of reaching the United States. The head of NATO on Tuesday urged all United Nations members to fully and transparently implement sanctions against North Korea.  North Korea s ballistic and nuclear tests are an affront to the United Nations Security Council,  NATO Secretary General Jens Stoltenberg said in Tokyo, where he met Japanese Prime Minister Shinzo Abe. Separately, a South Korean lawmaker said North Korea probably stole South Korean warship blueprints after hacking into a local shipbuilder s database last April. Expectations had been growing for a warming in the frosty bilateral ties following this month s conclave of China s Communist Party, during which Xi cemented his status as China s most powerful leader after Mao Zedong. Earlier this month, South Korea and China agreed to renew a $56 billion currency swap agreement, while Chinese airlines are reportedly planning to restore flight routes to South Korea that had been cut during the spat. Tuesday s agreement came after high-level talks led by Nam Gwan-pyo, deputy director of national security of the Blue House, and Kong Xuanyou, assistant foreign minister of China and the country s special envoy for North Korea-related matters. "}
    
]

test_vectors = np.array([vectorize_and_combine(item, doc2vec_model) for item in test_data])

predictions = classifier_model.predict(test_vectors)
print("Predictions:", predictions)


# In[ ]:





# In[ ]:





# In[ ]:




