from sklearn import svm
import threading
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import time
import matplotlib
from sklearn.metrics import precision_score, recall_score

matplotlib.use('Agg')

def save_commonplot(wordcloud, i):
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Common Words in Cluster {i+1}")
    plt.savefig(f"static/common_words_{i}.png")
    plt.clf()

def save_uniqueplot(wordcloud, i):
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Unique Words in Cluster {i+1}")
    plt.savefig(f"static/unique_words_{i}.png")
    plt.clf()

def naivebayes_model(df):
    text_data = df.iloc[:, 0]
    labels = df.iloc[:, 1]

    X_train, X_test, y_train, y_test = train_test_split(text_data, labels, test_size=0.1, random_state=42)

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    clf = MultinomialNB()
    clf.fit(X_train, y_train)
 
    joblib.dump(clf, 'naivebayes_model.pkl')
    vectorizer_filename = 'nbvectorizer.pkl'
    joblib.dump(vectorizer, vectorizer_filename)
    
    y_pred = clf.predict(X_test)


    classification = classification_report(y_test, y_pred, output_dict=True)
    accuracy = classification['accuracy']
    recall = classification['macro avg']['recall']
    precision = classification['macro avg']['precision']
    cm = confusion_matrix(y_test, y_pred)

    return accuracy, recall, precision, cm



def decisiontree_model(df):
    text_data = df.iloc[:, 0]
    labels = df.iloc[:, 1]

    X_train, X_test, y_train, y_test = train_test_split(text_data, labels, test_size=0.1, random_state=42)

    # Vectorize your data
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train your decision tree model
    clf = DecisionTreeClassifier()
    clf.fit(X_train_vec, y_train)

    joblib.dump(clf, 'decisiontree_model.pkl')
    vectorizer_filename = 'dtvectorizer.pkl'
    joblib.dump(vectorizer, vectorizer_filename)

    # Evaluate your decision tree model
    y_pred = clf.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    return accuracy, recall, precision,cm

def svm_model(df):
    text_data = df.iloc[:, 0]
    labels = df.iloc[:, 1]
    
    # Vectorize text data
    vectorizer = TfidfVectorizer()
    vectorized_data = vectorizer.fit_transform(text_data)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(vectorized_data, labels, test_size=0.1, random_state=42)

    # Train SVM classifier
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)

    joblib.dump(clf, 'svm_model.pkl')
    vectorizer_filename = 'svmvectorizer.pkl'
    joblib.dump(vectorizer, vectorizer_filename)

    # Predict test set
    print(np.shape(X_test))
    y_pred = clf.predict(X_test)

    classification = classification_report(y_test, y_pred, output_dict=True)
    accuracy = classification['accuracy']
    recall = classification['macro avg']['recall']
    precision = classification['macro avg']['precision']
    cm = confusion_matrix(y_test, y_pred)


    return accuracy, recall, precision,cm
    
def knn_model(df):
    text_data = df.iloc[:, 0]
    labels = df.iloc[:, 1]

    # Vectorize text data
    vectorizer = TfidfVectorizer()
    vectorized_data = vectorizer.fit_transform(text_data)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(vectorized_data, labels, test_size=0.1, random_state=42)

    # Create a KNN classifier object with k=5
    knn = KNeighborsClassifier(n_neighbors=3)

    # Train the classifier on the training set
    knn.fit(X_train, y_train)

    joblib.dump(knn, 'knn_model.pkl')
    vectorizer_filename = 'knnvectorizer.pkl'
    joblib.dump(vectorizer, vectorizer_filename)

    # Make predictions on the testing set
    y_pred = knn.predict(X_test)

    classification = classification_report(y_test, y_pred, output_dict=True)
    accuracy = classification['accuracy']
    recall = classification['macro avg']['recall']
    precision = classification['macro avg']['precision']
    cm = confusion_matrix(y_test, y_pred)

    return accuracy, recall, precision,cm

def kmean_model(df):
    text_data = df.iloc[:, 0]

    # Convert text data to TF-IDF vectors
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(text_data)

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=5, n_init=50)
    labels = kmeans.fit_predict(X)
    
    joblib.dump(kmeans, 'kmean_model.pkl')
    vectorizer_filename = 'kmeanvectorizer.pkl'
    joblib.dump(vectorizer, vectorizer_filename)

    # Get common words and unique words in each cluster
    common_words = []
    unique_words = []
    for i in range(5):
        cluster_words = []
        for j in range(len(text_data)):
            if labels[j] == i:
                cluster_words += text_data[j].split()
        common_words.append(set(cluster_words))
        unique_cluster_words = set(cluster_words) - set().union(*common_words)
        unique_words.append(unique_cluster_words)
        common_words[-1] = common_words[-1].union(unique_cluster_words)

    silhouette_scoree = silhouette_score(X, labels)

    # Generate word clouds for each cluster
    for i in range(5):
        if common_words[i]:
            wordcloud = WordCloud(background_color="white").generate(' '.join(common_words[i]))
            save_commonplot(wordcloud, i)
            # time.sleep(5)
        if unique_words[i]:
            wordcloud = WordCloud(background_color="white").generate(' '.join(unique_words[i]))
            save_uniqueplot(wordcloud, i)    
        cm = confusion_matrix(y_test, y_pred)

    return text_data, labels, silhouette_scoree ,cm

def dbscan_model(df):
    data = df.iloc[:, 0]

    # Vectorize data
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data)

    # Cluster data using DBSCAN
    dbscan = DBSCAN(eps=1, min_samples=7)
    dbscan.fit(X)

    joblib.dump(dbscan, 'dbscan_model.pkl')
    vectorizer_filename = 'dbscanvectorizer.pkl'
    joblib.dump(vectorizer, vectorizer_filename)

    # Get cluster assignments
    labels = dbscan.labels_

    # Get silhouette score
    silhouette_avg = silhouette_score(X, labels)

    # Get word clouds for each cluster
    wordclouds = []
    for label in set(labels):
        if label == -1:
            continue
        cluster_data = [data[i] for i, l in enumerate(labels) if l == label]
        cluster_X = vectorizer.transform(cluster_data)
        wordcloud = WordCloud(background_color="white").generate(" ".join(cluster_data))
        wordclouds.append((label, wordcloud.to_svg()))

    # Get common and unique words for each cluster
    common_words = {}
    unique_words = {}
    for label in set(labels):
        if label == -1:
            continue
        cluster_data = [data[i] for i, l in enumerate(labels) if l == label]
        cluster_X = vectorizer.transform(cluster_data)
        feature_names = vectorizer.get_feature_names_out()
        tfidf_sum = cluster_X.sum(axis=0)
        tfidf_scores = [(feature_names[i], tfidf_sum[0, i]) for i in range(len(feature_names))]
        tfidf_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
        common_words[label] = tfidf_scores[:5]
        unique_words[label] = [w for w, s in tfidf_scores[-5:]]

    # Assign labels to clusters
    cluster_labels = {}
    for label in set(labels):
        if label == -1:
            cluster_labels[label] = "Noise"
        else:
            cluster_labels[label] = f"Cluster {label}"

    return labels, silhouette_avg, wordclouds, common_words, unique_words, cluster_labels

df = pd.read_excel("train data.xlsx")

dbscan_model(df)

