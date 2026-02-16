import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import plotly.graph_objects as go

# =========================================
# PAGE CONFIGURATION
# =========================================
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
    }
    </style>
""", unsafe_allow_html=True)

# =========================================
# TITLE & DESCRIPTION
# =========================================
st.markdown('<p class="main-header">🛍️ Customer Segmentation Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">K-Means Clustering Analysis for Mall Customers</p>', unsafe_allow_html=True)

st.markdown("---")

# =========================================
# SIDEBAR
# =========================================
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Select Page:",
    ["🏠 Home", "📈 Data Exploration", "🤖 Clustering Analysis", "🎯 Predictions", "ℹ️ About"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 👨‍💻 Project Info")
st.sidebar.info("""
**Author:** Aliaa Mohamed  
**Date:** February 2026  
**Algorithm:** K-Means Clustering  
**Dataset:** Mall Customers
""")

# =========================================
# LOAD DATA
# =========================================
@st.cache_data
def load_data():
    df = pd.read_csv('data/Mall_Customers.csv')
    return df

@st.cache_resource
def load_models():
    try:
        kmeans_model = joblib.load('outputs/kmeans_model.pkl')
        scaler = joblib.load('outputs/scaler.pkl')
        return kmeans_model, scaler
    except:
        return None, None

df = load_data()
kmeans_model, scaler = load_models()

# =========================================
# PAGE 1: HOME
# =========================================
if page == "🏠 Home":
    st.header("🎯 Project Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", len(df), delta="200 records")
    with col2:
        st.metric("Features Used", "3", delta="Age, Income, Spending")
    with col3:
        st.metric("Clusters", "5", delta="Optimal K")
    with col4:
        st.metric("Algorithm", "K-Means", delta="Unsupervised")
    
    st.markdown("---")
    
    # Introduction
    st.subheader("📌 What is Customer Segmentation?")
    st.write("""
    Customer segmentation is the process of dividing customers into groups based on common characteristics 
    so companies can market to each group effectively and appropriately.
    
    **In this project:**
    - We use **K-Means Clustering** to segment mall customers
    - Based on **Age**, **Annual Income**, and **Spending Score**
    - To identify **5 distinct customer groups**
    - For targeted marketing strategies
    """)
    
    st.markdown("---")
    
    # Dataset Preview
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.markdown("---")
    
    # Key Insights
    st.subheader("💡 Key Insights")
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **🎯 Target Segments Identified:**
        - Premium Customers (High Income, High Spending)
        - Budget Shoppers (Low Income, Low Spending)
        - Conservative Spenders (High Income, Low Spending)
        - Impulse Buyers (Low Income, High Spending)
        - Practical Shoppers (Medium Income, Medium Spending)
        """)
    
    with col2:
        st.info("""
        **📈 Business Applications:**
        - Personalized marketing campaigns
        - Product recommendations
        - Pricing strategies
        - Customer retention programs
        - Resource allocation
        """)

# =========================================
# PAGE 2: DATA EXPLORATION
# =========================================
elif page == "📈 Data Exploration":
    st.header("📈 Exploratory Data Analysis")
    
    # Dataset Statistics
    st.subheader("📊 Dataset Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Dataset Info:**")
        st.write(df.describe())
    
    with col2:
        st.write("**Missing Values:**")
        st.write(df.isnull().sum())
    
    st.markdown("---")
    
    # Distribution Plots
    st.subheader("📊 Feature Distributions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df['Age'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Age')
        ax.set_ylabel('Frequency')
        ax.set_title('Age Distribution')
        ax.grid(alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df['Annual Income (k$)'], bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Annual Income (k$)')
        ax.set_ylabel('Frequency')
        ax.set_title('Income Distribution')
        ax.grid(alpha=0.3)
        st.pyplot(fig)
    
    with col3:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df['Spending Score (1-100)'], bins=20, color='salmon', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Spending Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Spending Distribution')
        ax.grid(alpha=0.3)
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Scatter Plot (Interactive)
    st.subheader("🎨 Income vs Spending Score (Interactive)")
    
    fig = px.scatter(
        df, 
        x='Annual Income (k$)', 
        y='Spending Score (1-100)',
        color='Gender',
        size='Age',
        hover_data=['CustomerID', 'Age'],
        title='Annual Income vs Spending Score',
        color_discrete_map={'Male': '#1E88E5', 'Female': '#E91E63'}
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Gender Distribution
    st.subheader("👥 Gender Distribution")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        gender_counts = df['Gender'].value_counts()
        st.write(gender_counts)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=['#1E88E5', '#E91E63'], startangle=90)
        ax.set_title('Gender Distribution')
        st.pyplot(fig)

# =========================================
# PAGE 3: CLUSTERING ANALYSIS
# =========================================
elif page == "🤖 Clustering Analysis":
    st.header("🤖 K-Means Clustering Analysis")
    
    # Elbow Method
    st.subheader("📐 Elbow Method - Finding Optimal K")
    
    if st.checkbox("Show Elbow Analysis"):
        with st.spinner("Calculating..."):
            features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
            X = df[features].values
            scaler_temp = StandardScaler()
            X_scaled = scaler_temp.fit_transform(X)
            
            inertia_values = []
            silhouette_scores = []
            K_range = range(2, 11)
            
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X_scaled)
                inertia_values.append(kmeans.inertia_)
                silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(K_range, inertia_values, marker='o', linewidth=2, markersize=8, color='steelblue')
                ax.axvline(x=5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Optimal K=5')
                ax.set_xlabel('Number of Clusters (K)')
                ax.set_ylabel('Inertia')
                ax.set_title('Elbow Method')
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(K_range, silhouette_scores, marker='s', linewidth=2, markersize=8, color='darkorange')
                ax.set_xlabel('Number of Clusters (K)')
                ax.set_ylabel('Silhouette Score')
                ax.set_title('Silhouette Analysis')
                ax.grid(alpha=0.3)
                st.pyplot(fig)
    
    st.markdown("---")
    
    # Cluster Visualization
    st.subheader("🎨 Cluster Visualization")
    
    if kmeans_model is not None:
        features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
        X = df[features].values
        X_scaled = scaler.transform(X)
        df['Cluster'] = kmeans_model.predict(X_scaled)
        
        # 2D Visualization (Interactive)
        st.write("**2D View: Income vs Spending**")
        fig = px.scatter(
            df, 
            x='Annual Income (k$)', 
            y='Spending Score (1-100)',
            color='Cluster',
            hover_data=['CustomerID', 'Age'],
            title='Customer Segments (2D)',
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # 3D Visualization (Interactive)
        st.write("**3D View: Age, Income, Spending**")
        fig = px.scatter_3d(
            df,
            x='Age',
            y='Annual Income (k$)',
            z='Spending Score (1-100)',
            color='Cluster',
            hover_data=['CustomerID', 'Gender'],
            title='Customer Segments (3D)',
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Cluster Statistics
        st.subheader("📊 Cluster Characteristics")
        cluster_stats = df.groupby('Cluster')[features].mean().round(2)
        st.dataframe(cluster_stats, use_container_width=True)
        
        # Cluster Distribution
        st.subheader("📈 Cluster Distribution")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            cluster_counts = df['Cluster'].value_counts().sort_index()
            st.write(cluster_counts)
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 5))
            cluster_counts.plot(kind='bar', color='teal', edgecolor='black', alpha=0.7, ax=ax)
            ax.set_xlabel('Cluster')
            ax.set_ylabel('Number of Customers')
            ax.set_title('Customer Distribution Across Clusters')
            ax.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=0)
            st.pyplot(fig)
        
        st.markdown("---")
        
        # Cluster Insights
        st.subheader("💡 Cluster Insights")
        
        interpretations = {
            0: ("Budget Shoppers", "Young, Low Income, Low Spending", "🛒"),
            1: ("Premium Customers (VIP)", "Middle-aged, High Income, High Spending", "💎"),
            2: ("Conservative Spenders", "Young, High Income, Low Spending", "💰"),
            3: ("Impulse Buyers", "Middle-aged, Low Income, High Spending", "🛍️"),
            4: ("Practical Shoppers", "Older, Medium Income, Medium Spending", "🎯")
        }
        
        for cluster_id in sorted(df['Cluster'].unique()):
            if cluster_id in interpretations:
                title, desc, emoji = interpretations[cluster_id]
                st.info(f"{emoji} **Cluster {cluster_id}: {title}**  \n{desc}")
    
    else:
        st.warning("⚠️ Model not found. Please train the model first in the Jupyter Notebook.")

# =========================================
# PAGE 4: PREDICTIONS
# =========================================
elif page == "🎯 Predictions":
    st.header("🎯 Predict Customer Segment")
    
    st.write("Enter customer information to predict their segment:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider("Age", 18, 70, 30)
    with col2:
        income = st.slider("Annual Income (k$)", 15, 140, 60)
    with col3:
        spending = st.slider("Spending Score (1-100)", 1, 100, 50)
    
    if st.button("🔮 Predict Segment", type="primary"):
        if kmeans_model is not None and scaler is not None:
            # Prepare input
            input_data = np.array([[age, income, spending]])
            input_scaled = scaler.transform(input_data)
            
            # Predict
            cluster = kmeans_model.predict(input_scaled)[0]
            
            # Display result
            st.success(f"### 🎯 Predicted Cluster: **{cluster}**")
            
            interpretations = {
                0: ("Budget Shoppers", "Young, Low Income, Low Spending", "Focus on value and discounts"),
                1: ("Premium Customers (VIP)", "Middle-aged, High Income, High Spending", "Offer premium products and VIP treatment"),
                2: ("Conservative Spenders", "Young, High Income, Low Spending", "Promote savings programs and investment options"),
                3: ("Impulse Buyers", "Middle-aged, Low Income, High Spending", "Use limited-time offers and flash sales"),
                4: ("Practical Shoppers", "Older, Medium Income, Medium Spending", "Emphasize quality and practicality")
            }
            
            if cluster in interpretations:
                title, desc, strategy = interpretations[cluster]
                st.info(f"""
                **{title}**  
                _{desc}_  
                
                **💡 Marketing Strategy:**  
                {strategy}
                """)
            
            # Visualize on scatter plot
            st.subheader("📍 Customer Position on Map")
            
            # ✅ FIX: Create Cluster column if not exists
            if 'Cluster' not in df.columns:
                features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
                X = df[features].values
                X_scaled = scaler.transform(X)
                df['Cluster'] = kmeans_model.predict(X_scaled)
            
            fig = px.scatter(
                df, 
                x='Annual Income (k$)', 
                y='Spending Score (1-100)',
                color='Cluster',
                hover_data=['CustomerID', 'Age'],
                title='Your Customer Position',
                color_continuous_scale='viridis',
                opacity=0.6
            )
            
            # Add the new customer
            fig.add_scatter(
                x=[income],
                y=[spending],
                mode='markers',
                marker=dict(size=20, color='red', symbol='star', line=dict(width=2, color='white')),
                name='New Customer'
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.error("⚠️ Model not loaded. Please train the model first.")
# =========================================
# PAGE 5: ABOUT
# =========================================
elif page == "ℹ️ About":
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ### 🎯 Project Objective
    This project demonstrates **Customer Segmentation** using **K-Means Clustering** algorithm.
    The goal is to segment mall customers into distinct groups based on their characteristics.
    
    ### 📊 Dataset
    - **Source:** Mall Customers Dataset
    - **Size:** 200 customers
    - **Features:** CustomerID, Gender, Age, Annual Income, Spending Score
    
    ### 🧠 Methodology
    1. **Data Exploration:** Analyze distributions and relationships
    2. **Data Preprocessing:** Standardize features for K-Means
    3. **Optimal K Selection:** Use Elbow Method and Silhouette Score
    4. **Model Training:** Train K-Means with K=5
    5. **Visualization:** 2D and 3D cluster visualization
    6. **Interpretation:** Identify customer segments and strategies
    
    ### 🛠️ Technologies Used
    - **Python** - Programming Language
    - **Pandas** - Data Manipulation
    - **Scikit-learn** - Machine Learning
    - **Matplotlib & Seaborn** - Static Visualization
    - **Plotly** - Interactive Visualization
    - **Streamlit** - Web Application
    
    ### 📈 Key Results
    - **5 Customer Segments** identified
    - **Clear patterns** in spending behavior
    - **Actionable insights** for marketing strategies
    
    ### 👨‍💻 Author
    **Aliaa Mohamed**  
    Data Science Enthusiast  
    February 2026
    
    ### 🔗 Connect
    - LinkedIn: [www.linkedin.com/in/aliaa-mohamed-abdo]
    - GitHub: [https://github.com/Aliaa-mohamed47]
    - Email: aliaamohamed472005@gmail.com
    """)
    
    st.markdown("---")
    
    st.subheader("📚 References")
    st.markdown("""
    - [K-Means Clustering - Scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#k-means)
    - [Streamlit Documentation](https://docs.streamlit.io/)
    - [Customer Segmentation Guide](https://www.kaggle.com/datasets)
    """)

# =========================================
# FOOTER
# =========================================
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with ❤️ using Streamlit | © 2026 Aliaa Mohamed</p>
    </div>
""", unsafe_allow_html=True)