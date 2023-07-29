import math
from zipfile import ZipFile
from flask import Flask, flash, render_template, request, send_file, redirect
import pandas as pd
from preprocess import preprocess_data
from model_training import svm_model, knn_model, decisiontree_model, naivebayes_model, kmean_model, dbscan_model
from model_testing import svm_test, naivebayes_test, decisiontree_test, knn_test, kmean_test, dbscan_test
import mysql.connector
from math import ceil
import pdb
from sklearn.metrics import confusion_matrix
from flask import Flask, render_template, request, redirect, flash, session
import secrets


app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # 16 bytes will generate a 32-character secret key


# Replace the placeholders with your MySQL database credentials
db = mysql.connector.connect(
    host="localhost",#ths is same
    user="root",
    password="a123",
    database="login"
)

@app.route('/login',methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']

        cursor = db.cursor()
        # Replace 'users' with the name of your table
        cursor.execute(f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}' AND role = '{role}'")
        user = cursor.fetchone()

        if user:
            if role == 'user':
                # Redirect to the user page
                return redirect('/user')
            elif role == 'admin':
                # Redirect to the admin page
                return redirect('/admin')
        else:
            # Redirect to an error page
            flash('Invalid username, password, or role. Please try again.', 'error')
            return redirect('/login')

    return render_template('login.html')

@app.route("/")
@app.route("/home")
def index():
    return render_template('login.html')


@app.route("/signuped", methods=['GET', 'POST'])
def signuped():
    return render_template('signup.html')
@app.route("/signup", methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # Get form data
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']
        cursor = db.cursor()
        # Insert the form data into the database
        query = "INSERT INTO users (username, password, role) VALUES (%s, %s, %s)"
        values = (username, password, role)
        cursor.execute(query, values)
        db.commit()
    return render_template('login.html')

@app.route("/admin")
def admin():
    return render_template('admin.html')

@app.route("/user")
def user():
    return render_template('user.html')


@app.route("/<models>")
def handle_models(models):
    selected_models = models.split(',')
    preprocessed_data_ = preprocessed_data

    # Initialize result variables
    accuracy = {}
    precision = {}
    recall = {}
    confusion_matrix = {}
    # Iterate over selected models and execute corresponding functions
    for model in selected_models:
        if model == "naivebayes":
            accuracy[model], precision[model], recall[model], confusion_matrix[model] = naivebayes_model(preprocessed_data_)
        elif model == "knn":
            accuracy[model], precision[model], recall[model], confusion_matrix[model] = knn_model(preprocessed_data_)
        elif model == "decisiontree":
            accuracy[model], precision[model], recall[model], confusion_matrix[model] = decisiontree_model(preprocessed_data_)
        elif model == "svm":
            accuracy[model], precision[model], recall[model], confusion_matrix[model] = svm_model(preprocessed_data_)

    # Pass the results to the template based on the selected models
    return render_template(
        'multi_models.html',
        selected_models=selected_models,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        confusion_matrix=confusion_matrix,
    )
# @app.route("/Cmodels")
# def Cmodels():
#     selected_models = request.args.get('models')
#     preprocessed_data_ = preprocessed_data

#     # text_data = []
#     # labels = []
#     # silhouette_score = 0.0
#     Labels = []
#     silhouette_avg = 0.0
#     wordclouds = []
#     common_words = []
#     unique_words = []
#     cluster_labels = []

#     # Iterate over selected models and execute corresponding functions
#     for model in selected_models:
#         # if model == "kmean":
#         #      text_data[model], labels[model], silhouette_score[model] =  kmean_model(preprocessed_data)     
#         if model == "dbscan":
#             Labels, silhouette_avg, wordclouds, common_words, unique_words, cluster_labels = dbscan_model(preprocessed_data)
#     # Pass the results to the template based on the selected models
#     return render_template(
#         'Cmodels.html',
#         # selected_models=selected_models,
#         # text_data = text_data, 
#         # labels = labels, 
#         # silhouette_score = silhouette_score, 
#         optimal_n_clusters = 5,
#         Labels=Labels, 
#         silhouette_avg=silhouette_avg, 
#         wordclouds=wordclouds, 
#         common_words=common_words, 
#         nique_words=unique_words, 
#         cluster_labels=cluster_labels
#         )
@app.route('/Cmodels', methods=['GET', 'POST'])
def c_model():
    if request.method == 'POST':
        selected_models = request.form.getlist('model')

        # Load data
        df = preprocessed_data  # Replace with your own data loading logic

        results = {}
        if 'kmean' in selected_models:
            # Run K-means model
            text_data, labels, silhouette_scoree = kmean_model(df)
            results['kmeans'] = {
                'text_data': text_data,
                'labels': labels,
                'silhouette_scoree': silhouette_scoree
            }


        if 'dbscan' in selected_models:
            # Run DBSCAN model
            dbscan_labels, dbscan_silhouette_avg, dbscan_wordclouds, dbscan_common_words, dbscan_unique_words, dbscan_cluster_labels = dbscan_model(df)
            results['dbscan'] = {
                'labels': dbscan_labels,
                'silhouette_avg': dbscan_silhouette_avg,
                'wordclouds': dbscan_wordclouds,
                'common_words': dbscan_common_words,
                'unique_words': dbscan_unique_words,
                'cluster_labels': dbscan_cluster_labels
            }

    return render_template('Cmodels.html', results=results)
# @app.route('/preprocess', methods=['POST'])
# def upload_file():
#     # Get the uploaded file from the request object
#     uploaded_file = request.files['dataset']
#     options = request.form.getlist('preprocessoption')
#     global preprocessed_data 
#     preprocessed_data = preprocess_data(uploaded_file, options)
#     html_df = preprocessed_data.to_html()
#     # preprocess(uploaded_file)
#     return render_template('admin.html', dataframe = html_df)
#working
@app.route('/preprocess', methods=['POST'])
def upload_file():
    # Get the uploaded file from the request object
    uploaded_file = request.files['dataset']
    options = request.form.getlist('preprocessoption')
    global preprocessed_data
    preprocessed_data = preprocess_data(uploaded_file, options)
    rows = preprocessed_data.values.tolist()
    columns = preprocessed_data.columns.tolist()
    return render_template('admin.html', rows=rows, columns=columns)

# @app.route('/preprocess', methods=['POST', 'GET'])
# def preprocess():
#     if request.method == 'POST':
#         # Get the uploaded file from the request object
#         uploaded_file = request.files['dataset']
#         options = request.form.getlist('preprocessoption')
#         glob preprocessed_data
#         preprocealssed_data = preprocess_data(uploaded_file, options)
        
#     # Retrieve the current page number from the query parameters
#     page = request.args.get('page', default=1, type=int)
#     rows_per_page = 10  # Number of rows to display per page
    
#     # Calculate the start and end indices for the rows to display on the current page
#     start_index = (page - 1) * rows_per_page
#     end_index = start_index + rows_per_page
    
#     # Get the subset of rows to display on the current page
#     rows = preprocessed_data.iloc[start_index:end_index].values.tolist()
#     columns = preprocessed_data.columns.tolist()
    
#      # Calculate the total number of pages
#     total_rows = len(preprocessed_data)
#     num_pages = math.ceil(total_rows / rows_per_page)
    
#     return render_template('admin.html', rows=rows, columns=columns, current_page=page, num_pages=num_pages)













@app.route('/userpreprocess', methods=['POST'])
def userupload_file():
    # Get the uploaded file from the request object
    uploaded_file = request.files['dataset']
    global test_data 
    test_data = pd.read_excel(uploaded_file)
    
    return render_template('usermodel.html')

@app.route('/results', methods=['POST'])
def results():
    selected_models = []
    dataframes = {}

    if 'nb' in request.form:
        selected_models.append('Naive Bayes')
        dataframe_nb = naivebayes_test(test_data)
        dataframe_nb.to_excel("Predicted_data_naivebayes.xlsx")
        dataframes['naivebayes'] = dataframe_nb

    if 'svm' in request.form:
        selected_models.append('SVM')
        dataframe_svm = svm_test(test_data)
        dataframe_svm.to_excel("Predicted_data_SVM.xlsx")
        dataframes['svm'] = dataframe_svm

    if 'knn' in request.form:
        selected_models.append('k-Nearest Neighbor')
        dataframe_knn = knn_test(test_data)
        dataframe_knn.to_excel("Predicted_data_kNN.xlsx")
        dataframes['knn'] = dataframe_knn

    if 'decisiontree' in request.form:
        selected_models.append('Decision Trees')
        dataframe_dt = decisiontree_test(test_data)
        dataframe_dt.to_excel("Predicted_data_DT.xlsx")
        dataframes['decisiontree'] = dataframe_dt

    if 'kmean' in request.form:
        selected_models.append('K-means')
        dataframe_km = kmean_test(test_data)
        dataframe_km.to_excel("Predicted_data_KM.xlsx")
        dataframes['kmean'] = dataframe_km

    if 'dbscan' in request.form:
        selected_models.append('DBSCAN')
        dataframe_db = dbscan_test(test_data)
        dataframe_db.to_excel("Predicted_data_DB.xlsx")
        dataframes['dbscan'] = dataframe_db

    return render_template('datapredictionNEW.html', models=selected_models, dataframes=dataframes)


@app.route('/download/<model>')
def download(model):
    file_path = f"Predicted_data_{model.replace(' ', '_')}.xlsx"
    return send_file(file_path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)