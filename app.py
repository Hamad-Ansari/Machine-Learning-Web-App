import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64
from PIL import Image
# Make sure librosa is installed: pip install librosa
import librosa
import warnings
warnings.filterwarnings('ignore')

# Function to load example datasets
def load_example_dataset(name):
    if name == 'Titanic':
        return pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
    elif name == 'Iris':
        return pd.read_csv('https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv')
    elif name == 'Tips':
        return sns.load_dataset('tips')
    elif name == 'Diamonds':
        return sns.load_dataset('diamonds')

# Function to handle missing values
def handle_missing_values(df, strategy='mean'):
    if strategy == 'mean':
        return df.fillna(df.mean())
    elif strategy == 'median':
        return df.fillna(df.median())
    elif strategy == 'mode':
        return df.fillna(df.mode().iloc[0])

# Function for EDA
def perform_eda(df):
    st.subheader("Exploratory Data Analysis")
    st.write("Data Shape:", df.shape)
    st.write("Data Types:")
    st.write(df.dtypes)
    st.write("Summary Statistics:")
    st.write(df.describe())
    st.write("Correlation Matrix:")
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        corr = numeric_df.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, ax=ax)
        st.pyplot(fig)

# Function to detect and handle outliers using IQR
def handle_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    return df

# Function for visualization
def visualize_data(df):
    st.subheader("Data Visualization")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    if len(numeric_cols) > 1:
        fig = px.scatter_matrix(df[numeric_cols])
        st.plotly_chart(fig)
    
    for col in numeric_cols:
        fig = px.histogram(df, x=col)
        st.plotly_chart(fig)
    
    for col in categorical_cols:
        fig = px.bar(df[col].value_counts())
        st.plotly_chart(fig)

# Function to create preprocessing pipeline
def create_preprocessing_pipeline(numeric_features, categorical_features, scaler='StandardScaler'):
    if scaler == 'StandardScaler':
        scaler_obj = StandardScaler()
    elif scaler == 'MinMaxScaler':
        scaler_obj = MinMaxScaler()
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', scaler_obj)
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

# Function to train models
def train_models(X_train, X_test, y_train, y_test, task_type, model_type):
    models = {}
    scores = {}
    
    if task_type == 'Supervised':
        if model_type == 'Regression':
            models['Linear Regression'] = LinearRegression()
            models['Decision Tree Regressor'] = DecisionTreeRegressor()
            models['Random Forest Regressor'] = RandomForestRegressor()
            models['SVR'] = SVR()
            
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                scores[name] = {
                    'R2 Score': r2_score(y_test, y_pred),
                    'MSE': mean_squared_error(y_test, y_pred),
                    'MAE': mean_absolute_error(y_test, y_pred)
                }
        
        elif model_type == 'Classification':
            models['Logistic Regression'] = LogisticRegression()
            models['Decision Tree Classifier'] = DecisionTreeClassifier()
            models['Random Forest Classifier'] = RandomForestClassifier()
            models['SVC'] = SVC()
            
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                scores[name] = {
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Classification Report': classification_report(y_test, y_pred),
                    'Confusion Matrix': confusion_matrix(y_test, y_pred)
                }
    
    elif task_type == 'Unsupervised':
        if model_type == 'Clustering':
            models['KMeans'] = KMeans(n_clusters=3)
            for name, model in models.items():
                model.fit(X_train)
                scores[name] = {'Inertia': model.inertia_}
    
    return models, scores

# Function to plot model scores
def plot_model_scores(scores, task_type):
    if task_type == 'Supervised':
        fig = go.Figure()
        for model, metrics in scores.items():
            if 'Accuracy' in metrics:
                fig.add_trace(go.Bar(x=[model], y=[metrics['Accuracy']], name=model))
        fig.update_layout(title='Model Accuracy Comparison', xaxis_title='Model', yaxis_title='Accuracy')
        st.plotly_chart(fig)

# Function to download results
def download_results(scores, format='csv'):
    df_scores = pd.DataFrame(scores).T
    if format == 'csv':
        csv = df_scores.to_csv()
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="model_scores.csv">Download CSV</a>'
        st.markdown(href, unsafe_allow_html=True)
    elif format == 'json':
        json_str = df_scores.to_json()
        b64 = base64.b64encode(json_str.encode()).decode()
        href = f'<a href="data:file/json;base64,{b64}" download="model_scores.json">Download JSON</a>'
        st.markdown(href, unsafe_allow_html=True)

# Main Streamlit app
def main():
    st.title("Machine Learning Web App")
    st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        color: #000000;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Upload Data", "Handle Missing Values", "EDA", "Handle Outliers", "Visualization", "Preprocessing", "Model Training", "Results"])
    
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    
    if page == "Upload Data":
        st.header("Upload Data or Select Example Dataset")
        option = st.radio("Choose option", ["Upload File", "Select Example"])
        
        if option == "Upload File":
            uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'json', 'wav', 'jpg', 'png'])
            if uploaded_file is not None:
                if uploaded_file.type == 'audio/wav':
                    # Handle audio file
                    y, sr = librosa.load(uploaded_file)
                    st.write("Audio loaded. Shape:", y.shape)
                    # For simplicity, convert to DataFrame with features
                    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                    st.session_state.df = pd.DataFrame(mfccs.T)
                elif uploaded_file.type in ['image/jpeg', 'image/png']:
                    # Handle image file
                    image = Image.open(uploaded_file)
                    st.image(image)
                    # For simplicity, convert to DataFrame with pixel values
                    img_array = np.array(image)
                    st.session_state.df = pd.DataFrame(img_array.flatten()).T
                else:
                    st.session_state.df = pd.read_csv(uploaded_file)
        
        elif option == "Select Example":
            dataset_name = st.selectbox("Select Dataset", ["Titanic", "Iris", "Tips", "Diamonds"])
            if st.button("Load Dataset"):
                st.session_state.df = load_example_dataset(dataset_name)
        
        if st.session_state.df is not None:
            st.write("Data Preview:")
            st.dataframe(st.session_state.df.head())
    
    elif page == "Handle Missing Values":
        if st.session_state.df is not None:
            st.header("Handle Missing Values")
            strategy = st.selectbox("Choose strategy", ["mean", "median", "mode"])
            if st.button("Apply"):
                st.session_state.df = handle_missing_values(st.session_state.df, strategy)
                st.success("Missing values handled!")
                st.dataframe(st.session_state.df.head())
        else:
            st.warning("Please upload data first.")
    
    elif page == "EDA":
        if st.session_state.df is not None:
            perform_eda(st.session_state.df)
        else:
            st.warning("Please upload data first.")
    
    elif page == "Handle Outliers":
        if st.session_state.df is not None:
            st.header("Handle Outliers")
            numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns
            col_to_handle = st.selectbox("Select column", numeric_cols)
            if st.button("Handle Outliers"):
                st.session_state.df = handle_outliers(st.session_state.df, col_to_handle)
                st.success("Outliers handled!")
                st.dataframe(st.session_state.df.head())
        else:
            st.warning("Please upload data first.")
    
    elif page == "Visualization":
        if st.session_state.df is not None:
            visualize_data(st.session_state.df)
        else:
            st.warning("Please upload data first.")
    
    elif page == "Preprocessing":
        if st.session_state.df is not None:
            st.header("Preprocessing")
            numeric_features = st.multiselect("Numeric Features", st.session_state.df.select_dtypes(include=[np.number]).columns)
            categorical_features = st.multiselect("Categorical Features", st.session_state.df.select_dtypes(include=['object']).columns)
            scaler = st.selectbox("Scaler", ["StandardScaler", "MinMaxScaler"])
            if st.button("Create Pipeline"):
                preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features, scaler)
                st.session_state.preprocessor = preprocessor
                st.success("Preprocessing pipeline created!")
        else:
            st.warning("Please upload data first.")
    
    elif page == "Model Training":
        if st.session_state.df is not None and 'preprocessor' in st.session_state:
            st.header("Model Training")
            task_type = st.selectbox("Task Type", ["Supervised", "Unsupervised"])
            if task_type == "Supervised":
                model_type = st.selectbox("Model Type", ["Regression", "Classification"])
                target_col = st.selectbox("Target Column", st.session_state.df.columns)
                X = st.session_state.df.drop(target_col, axis=1)
                y = st.session_state.df[target_col]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Apply preprocessing
                X_train_processed = st.session_state.preprocessor.fit_transform(X_train)
                X_test_processed = st.session_state.preprocessor.transform(X_test)
                
                if st.button("Train Models"):
                    models, scores = train_models(X_train_processed, X_test_processed, y_train, y_test, task_type, model_type)
                    st.session_state.models = models
                    st.session_state.scores = scores
                    st.success("Models trained!")
            
            elif task_type == "Unsupervised":
                model_type = st.selectbox("Model Type", ["Clustering"])
                X = st.session_state.df
                X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
                X_train_processed = st.session_state.preprocessor.fit_transform(X_train)
                X_test_processed = st.session_state.preprocessor.transform(X_test)
                
                if st.button("Train Models"):
                    models, scores = train_models(X_train_processed, X_test_processed, None, None, task_type, model_type)
                    st.session_state.models = models
                    st.session_state.scores = scores
                    st.success("Models trained!")
        else:
            st.warning("Please upload data and create preprocessing pipeline first.")
    
    elif page == "Results":
        if 'scores' in st.session_state:
            st.header("Model Results")
            for model, metrics in st.session_state.scores.items():
                st.subheader(model)
                for metric, value in metrics.items():
                    if isinstance(value, str):
                        st.text(metric + ":")
                        st.text(value)
                    else:
                        st.write(f"{metric}: {value}")
            
            plot_model_scores(st.session_state.scores, "Supervised")
            
            format_option = st.selectbox("Download Format", ["csv", "json"])
            if st.button("Download Results"):
                download_results(st.session_state.scores, format_option)
        else:
            st.warning("Please train models first.")

if __name__ == "__main__":
    main()
