import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import time
import altair as alt
import os
from PIL import Image
from io import BytesIO
import base64
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Advanced Data Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 30px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 25px;
        font-weight: bold;
        color: #333;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .card {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        background-color: #f9f9f9;
    }
    .info-text {
        font-size: 16px;
        line-height: 1.6;
    }
    .highlight {
        background-color: #f0f7ff;
        padding: 15px;
        border-left: 5px solid #1E88E5;
        margin: 20px 0;
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        padding: 20px;
        color: #666;
        font-size: 14px;
        border-top: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Application title
    st.markdown('<div class="main-header">Advanced Data Analysis Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.sidebar.image("https://www.svgrepo.com/show/374167/data-analysis.svg", width=100)
        st.sidebar.markdown("## Dashboard Controls")
        
        # Session state initialization for theme
        if 'theme' not in st.session_state:
            st.session_state.theme = 'light'
        
        app_mode = st.sidebar.selectbox(
            "Select App Mode",
            ["Home", "Data Upload & Exploration", "Data Visualization", "Machine Learning", "About"]
        )
        
        # Theme selector
        theme = st.sidebar.radio(
            "Choose Theme",
            ["Light", "Dark"],
            index=0 if st.session_state.theme == 'light' else 1
        )
        
        if theme == "Dark" and st.session_state.theme != 'dark':
            st.session_state.theme = 'dark'
            st.markdown("""
            <style>
                .stApp {
                    background-color: #121212;
                    color: #ffffff;
                }
                .card {
                    background-color: #1e1e1e;
                    color: #ffffff;
                }
                .sub-header {
                    color: #ffffff;
                }
                .highlight {
                    background-color: #253245;
                }
                .footer {
                    border-top: 1px solid #333;
                    color: #999;
                }
                .stTextInput>div>div>input, .stSelectbox>div>div>input {
                    background-color: #333;
                    color: white;
                }
            </style>
            """, unsafe_allow_html=True)
        elif theme == "Light" and st.session_state.theme != 'light':
            st.session_state.theme = 'light'
            st.markdown("""
            <style>
                .stApp {
                    background-color: #ffffff;
                    color: #000000;
                }
                .card {
                    background-color: #f9f9f9;
                    color: #000000;
                }
                .sub-header {
                    color: #333;
                }
                .highlight {
                    background-color: #f0f7ff;
                }
                .footer {
                    border-top: 1px solid #eee;
                    color: #666;
                }
            </style>
            """, unsafe_allow_html=True)
            
        st.sidebar.markdown("---")
        
    # Initialize session state for data storage
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'models' not in st.session_state:
        st.session_state.models = {}
    if 'predictions' not in st.session_state:
        st.session_state.predictions = {}
    
    # Navigation based on app mode selection
    if app_mode == "Home":
        display_home()
    elif app_mode == "Data Upload & Exploration":
        display_data_upload()
    elif app_mode == "Data Visualization":
        display_data_visualization()
    elif app_mode == "Machine Learning":
        display_machine_learning()
    elif app_mode == "About":
        display_about()
    
    # Footer
    st.markdown('<div class="footer">© 2025 Advanced Data Analysis Dashboard | Created with Streamlit</div>', unsafe_allow_html=True)

def display_home():
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Welcome to Advanced Data Analysis Dashboard</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-text">
        This interactive dashboard allows you to:
        <ul>
            <li>Upload and explore your datasets</li>
            <li>Generate comprehensive visualizations</li>
            <li>Build and evaluate machine learning models</li>
            <li>Export results and insights</li>
        </ul>
        
        Get started by selecting "Data Upload & Exploration" from the sidebar.
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Features showcase
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Key Features</div>', unsafe_allow_html=True)
        
        feature_col1, feature_col2 = st.columns(2)
        
        with feature_col1:
            st.markdown("#### 📊 Data Visualization")
            st.markdown("Interactive charts with Plotly and Matplotlib")
            
            st.markdown("#### 🤖 Machine Learning")
            st.markdown("Train models and make predictions")
        
        with feature_col2:
            st.markdown("#### 🔍 Data Exploration")
            st.markdown("Analyze data patterns and statistics")
            
            st.markdown("#### 📱 Responsive Design")
            st.markdown("Works on desktop and mobile devices")
            
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Quick Start</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight">
        <b>1.</b> Upload your CSV or Excel file<br>
        <b>2.</b> Explore data patterns<br>
        <b>3.</b> Create visualizations<br>
        <b>4.</b> Build ML models<br>
        <b>5.</b> Export your insights
        </div>
        """, unsafe_allow_html=True)
        
        # Sample dataset options
        st.markdown("### Try Sample Datasets")
        sample_dataset = st.selectbox(
            "Load a sample dataset",
            ["None", "Iris Dataset", "Boston Housing", "Wine Quality"]
        )
        
        if sample_dataset != "None":
            if sample_dataset == "Iris Dataset":
                from sklearn.datasets import load_iris
                data = load_iris()
                df.sort_values(date_col), x=date_col, y=value_col, color=group_col,
                            title=f"{value_col} over time (grouped by {group_col})",
                            color_discrete_sequence=px.colors.sequential.get(color_theme, px.colors.sequential.Viridis)
                        )
                    else:
                        fig = px.line(
                            df.sort_values(date_col), x=date_col, y=value_col, color=group_col,
                            title=f"{value_col} over time (grouped by {group_col})",
                            color_discrete_sequence=px.colors.sequential.get(color_theme, px.colors.sequential.Viridis)
                        )
                    fig.update_layout(xaxis_title="Date", yaxis_title=value_col)
                    st.plotly_chart(fig, use_container_width=True)
                
                elif ts_plot_type == "Area Plot":
                    fig = px.area(
                        df.sort_values(date_col), x=date_col, y=value_col,
                        title=f"{value_col} over time",
                        color_discrete_sequence=[color_theme]
                    )
                    fig.update_layout(xaxis_title="Date", yaxis_title=value_col)
                    st.plotly_chart(fig, use_container_width=True)
                
                elif ts_plot_type == "Bar Plot":
                    # Aggregate by month/year
                    time_unit = st.selectbox("Time Unit", ["Day", "Week", "Month", "Quarter", "Year"])
                    
                    if time_unit == "Day":
                        df['time_period'] = df[date_col].dt.date
                    elif time_unit == "Week":
                        df['time_period'] = df[date_col].dt.to_period('W').astype(str)
                    elif time_unit == "Month":
                        df['time_period'] = df[date_col].dt.to_period('M').astype(str)
                    elif time_unit == "Quarter":
                        df['time_period'] = df[date_col].dt.to_period('Q').astype(str)
                    else:  # Year
                        df['time_period'] = df[date_col].dt.year
                    
                    agg_df = df.groupby('time_period')[value_col].mean().reset_index()
                    
                    fig = px.bar(
                        agg_df, x='time_period', y=value_col,
                        title=f"Average {value_col} by {time_unit}",
                        color_discrete_sequence=[color_theme]
                    )
                    fig.update_layout(xaxis_title=time_unit, yaxis_title=f"Avg {value_col}")
                    st.plotly_chart(fig, use_container_width=True)
                
                elif ts_plot_type == "Decomposition":
                    from statsmodels.tsa.seasonal import seasonal_decompose
                    
                    # Need to set the index to the date column
                    ts_df = df[[date_col, value_col]].copy()
                    ts_df.set_index(date_col, inplace=True)
                    
                    # Check if data is regular
                    freq = st.selectbox("Frequency", ["Infer", "Daily", "Weekly", "Monthly"])
                    
                    if freq == "Infer":
                        freq_map = None
                    else:
                        freq_map = {'Daily': 'D', 'Weekly': 'W', 'Monthly': 'M'}
                        freq = freq_map[freq]
                    
                    try:
                        with st.spinner("Performing time series decomposition..."):
                            # Decompose the time series
                            decomposition = seasonal_decompose(ts_df[value_col], model='additive', period=12, extrapolate_trend='freq')
                            
                            # Create subplots
                            fig = make_subplots(
                                rows=4, cols=1,
                                subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
                                vertical_spacing=0.1
                            )
                            
                            # Add traces
                            fig.add_trace(go.Scatter(x=ts_df.index, y=decomposition.observed, mode='lines', name='Observed'), row=1, col=1)
                            fig.add_trace(go.Scatter(x=ts_df.index, y=decomposition.trend, mode='lines', name='Trend'), row=2, col=1)
                            fig.add_trace(go.Scatter(x=ts_df.index, y=decomposition.seasonal, mode='lines', name='Seasonal'), row=3, col=1)
                            fig.add_trace(go.Scatter(x=ts_df.index, y=decomposition.resid, mode='lines', name='Residual'), row=4, col=1)
                            
                            # Update layout
                            fig.update_layout(height=800, title_text=f"Time Series Decomposition of {value_col}")
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error performing decomposition: {str(e)}")
                        st.info("Time series decomposition requires regular time series data. Try using a different frequency or aggregating the data.")
            else:
                st.info("No date columns detected. Please ensure your dataset contains a valid date column.")
        
        elif viz_type == "Interactive Plots":
            st.write("### Interactive Visualizations")
            
            interactive_type = st.selectbox(
                "Select Interactive Plot Type",
                ["Scatterplot Matrix", "Parallel Coordinates", "Sunburst Chart", "Treemap"]
            )
            
            if interactive_type == "Scatterplot Matrix":
                if len(numeric_cols) >= 2:
                    selected_cols = st.multiselect("Select Columns (2-5 recommended)", numeric_cols, default=numeric_cols[:3])
                    
                    if len(selected_cols) >= 2:
                        color_col = st.selectbox("Color by (optional)", ["None"] + df.columns.tolist())
                        
                        if color_col != "None":
                            fig = px.scatter_matrix(
                                df, dimensions=selected_cols, color=color_col,
                                title="Interactive Scatterplot Matrix",
                                color_continuous_scale=color_theme if color_col in numeric_cols else None
                            )
                        else:
                            fig = px.scatter_matrix(
                                df, dimensions=selected_cols,
                                title="Interactive Scatterplot Matrix",
                                color_discrete_sequence=[color_theme]
                            )
                        fig.update_layout(height=700)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Please select at least 2 columns.")
                else:
                    st.info("At least 2 numerical columns are required for Scatterplot Matrix.")
            
            elif interactive_type == "Parallel Coordinates":
                if len(numeric_cols) >= 3:
                    selected_cols = st.multiselect("Select Columns (3-7 recommended)", numeric_cols, default=numeric_cols[:5])
                    
                    if len(selected_cols) >= 3:
                        color_col = st.selectbox("Color by", df.columns.tolist())
                        
                        fig = px.parallel_coordinates(
                            df, dimensions=selected_cols, color=color_col,
                            title="Parallel Coordinates Plot",
                            color_continuous_scale=color_theme if color_col in numeric_cols else None
                        )
                        fig.update_layout(height=600)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Please select at least 3 columns.")
                else:
                    st.info("At least 3 numerical columns are required for Parallel Coordinates.")
            
            elif interactive_type == "Sunburst Chart":
                if len(categorical_cols) >= 2:
                    path_cols = st.multiselect("Select Hierarchy (2-3 levels recommended)", categorical_cols, default=categorical_cols[:2])
                    
                    if len(path_cols) >= 1:
                        value_col = st.selectbox("Value Column", ["Count"] + numeric_cols)
                        
                        if value_col == "Count":
                            # Count records
                            fig = px.sunburst(
                                df, path=path_cols,
                                title="Sunburst Chart",
                                color_discrete_sequence=px.colors.sequential.get(color_theme, px.colors.sequential.Viridis)
                            )
                        else:
                            # Sum values
                            fig = px.sunburst(
                                df, path=path_cols, values=value_col,
                                title=f"Sunburst Chart ({value_col})",
                                color_discrete_sequence=px.colors.sequential.get(color_theme, px.colors.sequential.Viridis)
                            )
                        fig.update_layout(height=700)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Please select at least 1 column for the hierarchy.")
                else:
                    st.info("At least 2 categorical columns are required for Sunburst Chart.")
            
            elif interactive_type == "Treemap":
                if len(categorical_cols) >= 1:
                    path_cols = st.multiselect("Select Hierarchy (1-3 levels recommended)", categorical_cols, default=categorical_cols[:2])
                    
                    if len(path_cols) >= 1:
                        value_col = st.selectbox("Value Column", ["Count"] + numeric_cols)
                        
                        if value_col == "Count":
                            # Count records
                            fig = px.treemap(
                                df, path=path_cols,
                                title="Treemap",
                                color_discrete_sequence=px.colors.sequential.get(color_theme, px.colors.sequential.Viridis)
                            )
                        else:
                            # Sum values
                            fig = px.treemap(
                                df, path=path_cols, values=value_col,
                                title=f"Treemap ({value_col})",
                                color_discrete_sequence=px.colors.sequential.get(color_theme, px.colors.sequential.Viridis)
                            )
                        fig.update_layout(height=700)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Please select at least 1 column for the hierarchy.")
                else:
                    st.info("At least 1 categorical column is required for Treemap.")
        
        elif viz_type == "3D Visualization":
            st.write("### 3D Visualization")
            
            if len(numeric_cols) >= 3:
                plot_3d_type = st.selectbox(
                    "Select 3D Plot Type",
                    ["3D Scatter", "3D Surface"]
                )
                
                if plot_3d_type == "3D Scatter":
                    x_col = st.selectbox("X-axis", numeric_cols)
                    y_col = st.selectbox("Y-axis", [col for col in numeric_cols if col != x_col])
                    z_col = st.selectbox("Z-axis", [col for col in numeric_cols if col not in [x_col, y_col]])
                    
                    color_col = st.selectbox("Color by (optional)", ["None"] + df.columns.tolist())
                    
                    if color_col != "None":
                        fig = px.scatter_3d(
                            df, x=x_col, y=y_col, z=z_col, color=color_col,
                            title=f"3D Scatter Plot ({x_col}, {y_col}, {z_col})",
                            color_continuous_scale=color_theme if color_col in numeric_cols else None
                        )
                    else:
                        fig = px.scatter_3d(
                            df, x=x_col, y=y_col, z=z_col,
                            title=f"3D Scatter Plot ({x_col}, {y_col}, {z_col})",
                            color_discrete_sequence=[color_theme]
                        )
                    fig.update_layout(height=700)
                    st.plotly_chart(fig, use_container_width=True)
                
                elif plot_3d_type == "3D Surface":
                    # Need to create a grid for surface plot
                    st.info("3D Surface plot requires data in a grid format. Converting your data...")
                    
                    x_col = st.selectbox("X-axis", numeric_cols)
                    y_col = st.selectbox("Y-axis", [col for col in numeric_cols if col != x_col])
                    z_col = st.selectbox("Z-axis", [col for col in numeric_cols if col not in [x_col, y_col]])
                    
                    # Create grid from scattered data
                    x_range = np.linspace(df[x_col].min(), df[x_col].max(), 20)
                    y_range = np.linspace(df[y_col].min(), df[y_col].max(), 20)
                    
                    from scipy.interpolate import griddata
                    
                    X, Y = np.meshgrid(x_range, y_range)
                    Z = griddata(
                        (df[x_col], df[y_col]), df[z_col], 
                        (X, Y),
                        method='cubic'
                    )
                    
                    fig = go.Figure(data=[go.Surface(z=Z, x=x_range, y=y_range)])
                    fig.update_layout(
                        title=f"3D Surface Plot ({x_col}, {y_col}, {z_col})",
                        scene=dict(
                            xaxis_title=x_col,
                            yaxis_title=y_col,
                            zaxis_title=z_col
                        ),
                        height=700
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("At least 3 numerical columns are required for 3D visualization.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Visualization Gallery
    if 'export_viz' in st.session_state and st.session_state.export_viz:
        st.markdown('<div class="sub-header">Export Visualization</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        export_format = st.radio("Select Export Format", ["PNG", "HTML", "PDF", "SVG"])
        
        export_width = st.slider("Width (px)", 400, 1600, 1200)
        export_height = st.slider("Height (px)", 300, 1200, 800)
        
        if st.button("Generate Exportable Version"):
            st.success("✅ Visualization ready for export!")
            st.info(f"In a production environment, this would generate a {export_format} file for download.")
        
        st.markdown('</div>', unsafe_allow_html=True)

def display_machine_learning():
    st.markdown('<div class="sub-header">Machine Learning</div>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("Please upload data first in the 'Data Upload & Exploration' section.")
        return
    
    df = st.session_state.data
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("### Model Configuration")
        
        ml_task = st.selectbox(
            "Select Task",
            ["Classification", "Regression", "Clustering", "Dimensionality Reduction"]
        )
        
        # Get numerical and categorical columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        if ml_task in ["Classification", "Regression"]:
            target_col = st.selectbox("Select Target Variable", df.columns.tolist())
            feature_cols = st.multiselect(
                "Select Features", 
                [col for col in df.columns if col != target_col],
                default=[col for col in numeric_cols if col != target_col][:5]
            )
            
            test_size = st.slider("Test Set Size (%)", 10, 50, 20) / 100
            random_state = st.number_input("Random State", 0, 100, 42)
            
            model_type = st.selectbox(
                "Select Model",
                ["Random Forest", "Gradient Boosting", "Logistic Regression"] if ml_task == "Classification" else
                ["Random Forest", "Gradient Boosting", "Linear Regression"]
            )
        
        elif ml_task == "Clustering":
            feature_cols = st.multiselect(
                "Select Features", 
                df.columns.tolist(),
                default=numeric_cols[:5]
            )
            
            model_type = st.selectbox(
                "Select Clustering Algorithm",
                ["K-Means", "DBSCAN", "Hierarchical"]
            )
            
            n_clusters = st.slider("Number of Clusters", 2, 10, 3)
        
        elif ml_task == "Dimensionality Reduction":
            feature_cols = st.multiselect(
                "Select Features", 
                df.columns.tolist(),
                default=numeric_cols[:5]
            )
            
            model_type = st.selectbox(
                "Select Technique",
                ["PCA", "t-SNE", "UMAP"]
            )
            
            n_components = st.slider("Number of Components", 2, 10, 2)
        
        if st.button("Train Model"):
            st.session_state.train_model = True
        else:
            st.session_state.train_model = False
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        if ml_task in ["Classification", "Regression"] and 'train_model' in st.session_state and st.session_state.train_model:
            st.write(f"### {ml_task} Model: {model_type}")
            
            with st.spinner(f"Training {model_type} model..."):
                # Prepare the data
                X = df[feature_cols].copy()
                y = df[target_col].copy()
                
                # Handle categorical features
                categorical_features = [col for col in feature_cols if col in categorical_cols]
                
                if categorical_features:
                    st.info(f"Encoding categorical features: {', '.join(categorical_features)}")
                    
                    from sklearn.compose import ColumnTransformer
                    from sklearn.preprocessing import OneHotEncoder
                    
                    categorical_indices = [feature_cols.index(col) for col in categorical_features]
                    
                    preprocessor = ColumnTransformer(
                        transformers=[
                            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_indices)
                        ],
                        remainder='passthrough'
                    )
                    
                    X = preprocessor.fit_transform(X)
                
                # Split the data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
                
                # Scale the data
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train the model
                if ml_task == "Classification":
                    if model_type == "Random Forest":
                        model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
                    elif model_type == "Gradient Boosting":
                        model = GradientBoostingClassifier(n_estimators=100, random_state=random_state)
                    elif model_type == "Logistic Regression":
                        model = LogisticRegression(max_iter=1000, random_state=random_state)
                
                else:  # Regression
                    if model_type == "Random Forest":
                        from sklearn.ensemble import RandomForestRegressor
                        model = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
                    elif model_type == "Gradient Boosting":
                        from sklearn.ensemble import GradientBoostingRegressor
                        model = GradientBoostingRegressor(n_estimators=100, random_state=random_state)
                    elif model_type == "Linear Regression":
                        from sklearn.linear_model import LinearRegression
                        model = LinearRegression()
                
                # Fit the model
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                
                # Save model in session state
                st.session_state.models[model_type] = {
                    'model': model,
                    'scaler': scaler,
                    'preprocessor': preprocessor if categorical_features else None,
                    'features': feature_cols,
                    'target': target_col,
                    'task': ml_task
                }
                
                # Calculate metrics
                if ml_task == "Classification":
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    st.write(f"### Model Performance")
                    st.write(f"Accuracy: {accuracy:.4f}")
                    
                    # Classification report
                    report = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.write("Classification Report:")
                    st.dataframe(report_df, use_container_width=True)
                    
                    # Confusion Matrix
                    cm = confusion_matrix(y_test, y_pred)
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_title('Confusion Matrix')
                    st.pyplot(fig)
                    
                    # Feature importance
                    if model_type in ["Random Forest", "Gradient Boosting"]:
                        feature_importance = pd.DataFrame(
                            {'feature': feature_cols, 'importance': model.feature_importances_}
                        ).sort_values('importance', ascending=False)
                        
                        fig = px.bar(
                            feature_importance, x='feature', y='importance',
                            title='Feature Importance'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                else:  # Regression
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                    
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    st.write(f"### Model Performance")
                    st.write(f"R² Score: {r2:.4f}")
                    st.write(f"Mean Squared Error: {mse:.4f}")
                    st.write(f"Root Mean Squared Error: {rmse:.4f}")
                    st.write(f"Mean Absolute Error: {mae:.4f}")
                    
                    # Actual vs Predicted plot
                    fig = px.scatter(
                        x=y_test, y=y_pred,
                        labels={'x': 'Actual', 'y': 'Predicted'},
                        title='Actual vs Predicted'
                    )
                    fig.add_shape(
                        type='line', line=dict(dash='dash'),
                        x0=y_test.min(), y0=y_test.min(),
                        x1=y_test.max(), y1=y_test.max()
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature importance
                    if model_type in ["Random Forest", "Gradient Boosting"]:
                        feature_importance = pd.DataFrame(
                            {'feature': feature_cols, 'importance': model.feature_importances_}
                        ).sort_values('importance', ascending=False)
                        
                        fig = px.bar(
                            feature_importance, x='feature', y='importance',
                            title='Feature Importance'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Display model details
                st.write("### Model Details")
                st.json({
                    'Model Type': model_type,
                    'Features': feature_cols,
                    'Target': target_col,
                    'Test Size': test_size,
                    'Random State': random_state
                })
                
                # Save predictions
                st.session_state.predictions[model_type] = {
                    'actual': y_test,
                    'predicted': y_pred
                }
                
                # Download model
                if st.button("Download Model (Pickle)"):
                    st.info("In a production environment, this would download the model as a pickle file.")
        
        elif ml_task == "Clustering" and 'train_model' in st.session_state and st.session_state.train_model:
            st.write(f"### Clustering: {model_type}")
            
            with st.spinner(f"Running {model_type} clustering..."):
                # Prepare the data
                X = df[feature_cols].copy()
                
                # Handle categorical features
                categorical_features = [col for col in feature_cols if col in categorical_cols]
                
                if categorical_features:
                    st.info(f"Encoding categorical features: {', '.join(categorical_features)}")
                    
                    from sklearn.compose import ColumnTransformer
                    from sklearn.preprocessing import OneHotEncoder
                    
                    categorical_indices = [feature_cols.index(col) for col in categorical_features]
                    
                    preprocessor = ColumnTransformer(
                        transformers=[
                            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_indices)
                        ],
                        remainder='passthrough'
                    )
                    
                    X = preprocessor.fit_transform(X)
                
                # Scale the data
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Apply clustering
                if model_type == "K-Means":
                    from sklearn.cluster import KMeans
                    
                    model = KMeans(n_clusters=n_clusters, random_state=42)
                    clusters = model.fit_predict(X_scaled)
                    
                    # Silhouette score
                    from sklearn.metrics import silhouette_score
                    silhouette_avg = silhouette_score(X_scaled, clusters)
                    
                    st.write(f"Silhouette Score: {silhouette_avg:.4f}")
                
                elif model_type == "DBSCAN":
                    from sklearn.cluster import DBSCAN
                    
                    eps = st.slider("Epsilon (neighborhood size)", 0.1, 2.0, 0.5)
                    min_samples = st.slider("Min Samples", 3, 20, 5)
                    
                    model = DBSCAN(eps=eps, min_samples=min_samples)
                    clusters = model.fit_predict(X_scaled)
                    
                    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
                    st.write(f"Number of clusters found: {n_clusters}")
                    st.write(f"Number of noise points: {list(clusters).count(-1)}")
                
                elif model_type == "Hierarchical":
                    from sklearn.cluster import AgglomerativeClustering
                    
                    model = AgglomerativeClustering(n_clusters=n_clusters)
                    clusters = model.fit_predict(X_scaled)
                
                # Add clusters to dataframe
                df_clusters = df.copy()
                df_clusters['Cluster'] = clusters
                
                # Show cluster distribution
                cluster_dist = df_clusters['Cluster'].value_counts().reset_index()
                cluster_dist.columns = ['Cluster', 'Count']
                
                fig = px.bar(
                    cluster_dist, x='Cluster', y='Count',
                    title='Cluster Distribution',
                    color='Cluster'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Visualize clusters (only if we have at least 2 features)
                if len(feature_cols) >= 2:
                    # First get the first 2 PCA components
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_scaled)
                    
                    # Create a dataframe for plotting
                    pca_df = pd.DataFrame({
                        'PCA1': X_pca[:, 0],
                        'PCA2': X_pca[:, 1],
                        'Cluster': clusters
                    })
                    
                    fig = px.scatter(
                        pca_df, x='PCA1', y='PCA2', color='Cluster',
                        title='Cluster Visualization (PCA)',
                        color_continuous_scale='viridis' if model_type == 'DBSCAN' else None,
                        category_orders={'Cluster': sorted(pca_df['Cluster'].unique())}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show variance explained by PCA
                    st.write(f"Variance Explained by PCA: {sum(pca.explained_variance_ratio_):.2%}")
                    
                    # Analyze clusters
                    if model_type in ["K-Means", "Hierarchical"] or (model_type == "DBSCAN" and n_clusters > 0):
                        st.write("### Cluster Analysis")
                        
                        analysis_cols = st.multiselect(
                            "Select columns for analysis",
                            numeric_cols,
                            default=numeric_cols[:3]
                        )
                        
                        if analysis_cols:
                            # Compute cluster means
                            cluster_means = df_clusters.groupby('Cluster')[analysis_cols].mean()
                            
                            # Plot radar chart for each cluster
                            for cluster in sorted(df_clusters['Cluster'].unique()):
                                if cluster != -1:
                                    # Get the values for this cluster
                                    values = cluster_means.loc[cluster].values.tolist()
                                    
                                    # Complete the radar by repeating the first value
                                    values = values + [values[0]]
                                    
                                    # Create radar chart
                                    categories = analysis_cols + [analysis_cols[0]]
                                    
                                    fig = go.Figure()
                                    
                                    fig.add_trace(go.Scatterpolar(
                                        r=values,
                                        theta=categories,
                                        fill='toself',
                                        name=f'Cluster {cluster}'
                                    ))
                                    
                                    fig.update_layout(
                                        polar=dict(
                                            radialaxis=dict(visible=True)
                                        ),
                                        title=f"Profile for Cluster {cluster}"
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            # Show cluster statistics table
                            st.write("### Cluster Statistics")
                            st.dataframe(cluster_means, use_container_width=True)
                            
                            # Add download button for clustered data
                            if st.button("Download Clustered Data (CSV)"):
                                csv = df_clusters.to_csv(index=False)
                                b64 = base64.b64encode(csv.encode()).decode()
                                href = f'<a href="data:file/csv;base64,{b64}" download="clustered_data.csv">Download CSV File</a>'
                                st.markdown(href, unsafe_allow_html=True)
        
        elif ml_task == "Dimensionality Reduction" and 'train_model' in st.session_state and st.session_state.train_model:
            st.write(f"### Dimensionality Reduction: {model_type}")
            
            with st.spinner(f"Running {model_type}..."):
                # Prepare the data
                X = df[feature_cols].copy()
                
                # Handle categorical features
                categorical_features = [col for col in feature_cols if col in categorical_cols]
                
                if categorical_features:
                    st.info(f"Encoding categorical features: {', '.join(categorical_features)}")
                    
                    from sklearn.compose import ColumnTransformer
                    from sklearn.preprocessing import OneHotEncoder
                    
                    categorical_indices = [feature_cols.index(col) for col in categorical_features]
                    
                    preprocessor = ColumnTransformer(
                        transformers=[
                            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_indices)
                        ],
                        remainder='passthrough'
                    )
                    
                    X = preprocessor.fit_transform(X)
                
                # Scale the data
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Apply dimensionality reduction
                if model_type == "PCA":
                    model = PCA(n_components=n_components)
                    X_reduced = model.fit_transform(X_scaled)
                    
                    # Explained variance
                    explained_variance = model.explained_variance_ratio_
                    
                    st.write(f"### Explained Variance")
                    
                    # Create explained variance dataframe
                    explained_df = pd.DataFrame({
                        'Component': [f"PC{i+1}" for i in range(n_components)],
                        'Explained Variance (%)': explained_variance * 100,
                        'Cumulative Variance (%)': np.cumsum(explained_variance) * 100
                    })
                    
                    st.dataframe(explained_df, use_container_width=True)
                    
                    # Plot explained variance
                    fig = px.bar(
                        explained_df, x='Component', y='Explained Variance (%)',
                        title='Explained Variance by Component'
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=explained_df['Component'],
                            y=explained_df['Cumulative Variance (%)'],
                            mode='lines+markers',
                            name='Cumulative Variance',
                            yaxis='y2'
                        )
                    )
                    fig.update_layout(
                        yaxis2=dict(
                            title='Cumulative Variance (%)',
                            overlaying='y',
                            side='right'
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif model_type == "t-SNE":
                    from sklearn.manifold import TSNE
                    
                    perplexity = st.slider("Perplexity", 5, 50, 30)
                    learning_rate = st.slider("Learning Rate", 10, 1000, 200)
                    
                    model = TSNE(
                        n_components=n_components,
                        perplexity=perplexity,
                        learning_rate=learning_rate,
                        random_state=42
                    )
                    X_reduced = model.fit_transform(X_scaled)
                    
                    st.write("t-SNE does not provide explained variance information like PCA.")
                
                elif model_type == "UMAP":
                    import umap
                    
                    n_neighbors = st.slider("Number of Neighbors", 2, 100, 15)
                    min_dist = st.slider("Minimum Distance", 0.0, 1.0, 0.1)
                    
                    model = umap.UMAP(
                        n_components=n_components,
                        n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        random_state=42
                    )
                    X_reduced = model.fit_transform(X_scaled)
                    
                    st.write("UMAP does not provide explained variance information like PCA.")
                
                # Create a dataframe with the reduced data
                reduced_cols = [f"Component {i+1}" for i in range(n_components)]
                reduced_df = pd.DataFrame(X_reduced, columns=reduced_cols)
                
                # Add any categorical column for coloring (optional)
                if categorical_cols:
                    color_col = st.selectbox("Color by", ["None"] + categorical_cols)
                    
                    if color_col != "None":
                        reduced_df[color_col] = df[color_col].values
                
                # Visualize the reduced data
                if n_components >= 2:
                    st.write(f"### {model_type} Visualization")
                    
                    if n_components == 2:
                        if 'color_col' in locals() and color_col != "None":
                            fig = px.scatter(
                                reduced_df, x="Component 1", y="Component 2",
                                color=color_col,
                                title=f"{model_type} 2D Projection",
                                labels={"Component 1": "Component 1", "Component 2": "Component 2"}
                            )
                        else:
                            fig = px.scatter(
                                reduced_df, x="Component 1", y="Component 2",
                                title=f"{model_type} 2D Projection",
                                labels={"Component 1": "Component 1", "Component 2": "Component 2"}
                            )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif n_components >= 3:
                        if 'color_col' in locals() and color_col != "None":
                            fig = px.scatter_3d(
                                reduced_df, x="Component 1", y="Component 2", z="Component 3",
                                color=color_col,
                                title=f"{model_type} 3D Projection",
                                labels={
                                    "Component 1": "Component 1",
                                    "Component 2": "Component 2",
                                    "Component 3": "Component 3"
                                }
                            )
                        else:
                            fig = px.scatter_3d(
                                reduced_df, x="Component 1", y="Component 2", z="Component 3",
                                title=f"{model_type} 3D Projection",
                                labels={
                                    "Component 1": "Component 1",
                                    "Component 2": "Component 2",
                                    "Component 3": "Component 3"
                                }
                            )
                        fig.update_layout(height=700)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # If PCA, show component loadings
                    if model_type == "PCA":
                        st.write("### Component Loadings")
                        
                        loadings = model.components_
                        loading_df = pd.DataFrame(loadings, columns=feature_cols)
                        loading_df.index = [f"Component {i+1}" for i in range(n_components)]
                        
                        st.dataframe(loading_df, use_container_width=True)
                        
                        # Visualize loadings for top 2 components
                        st.write("### Loading Plot (First 2 Components)")
                        
                        loading_plot_df = pd.DataFrame({
                            'Feature': feature_cols * 2,
                            'Component': ['Component 1'] * len(feature_cols) + ['Component 2'] * len(feature_cols),
                            'Loading': np.concatenate([loadings[0], loadings[1]])
                        })
                        
                        fig = px.bar(
                            loading_plot_df, x='Feature', y='Loading', color='Component',
                            barmode='group',
                            title='Feature Loadings for Components 1 and 2'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Show reduced data table
                st.write("### Reduced Data (First 10 rows)")
                st.dataframe(reduced_df.head(10), use_container_width=True)
                
                # Add original features to the reduced dataframe
                result_df = pd.concat([df.reset_index(drop=True), reduced_df.reset_index(drop=True)], axis=1)
                
                # Add download button for reduced data
                if st.button("Download Reduced Data (CSV)"):
                    csv = result_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="reduced_data.csv">Download CSV File</a>'
                    st.markdown(href, unsafe_allow_html=True)
        
        else:
            if not ('train_model' in st.session_state and st.session_state.train_model):
                st.info(f"Configure your {ml_task} model and click 'Train Model' to get started.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ML predictions section
    if len(st.session_state.models) > 0:
        st.markdown('<div class="sub-header">Make Predictions</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        model_name = st.selectbox("Select Trained Model", list(st.session_state.models.keys()))
        
        model_info = st.session_state.models[model_name]
        model = model_info['model']
        scaler = model_info['scaler']
        preprocessor = model_info['preprocessor']
        features = model_info['features']
        
        st.write(f"### Make Predictions with {model_name}")
        st.write(f"This model was trained to predict: **{model_info['target']}**")
        st.write(f"Features required: {', '.join(features)}")
        
        # Choose input method
        input_method = st.radio("Input Method", ["Manual Input", "Upload File"])
        
        if input_method == "Manual Input":
            input_data = {}
            
            # Create input fields for each feature
            for feature in features:
                if feature in categorical_cols:
                    unique_values = df[feature].unique().tolist()
                    input_data[feature] = st.selectbox(f"Select {feature}", unique_values)
                else:
                    input_data[feature] = st.number_input(f"Enter {feature}", value=float(df[feature].mean()))
            
            if st.button("Make Prediction"):
                try:
                    # Create a dataframe from input
                    input_df = pd.DataFrame([input_data])
                    
                    # Preprocess if needed
                    if preprocessor is not None:
                        input_processed = preprocessor.transform(input_df)
                    else:
                        input_processed = input_df
                    
                    # Scale the input
                    input_scaled = scaler.transform(input_processed)
                    
                    # Make prediction
                    prediction = model.predict(input_scaled)[0]
                    
                    st.success(f"### Prediction: {prediction}")
                    
                    # If it's a classification model, show probabilities
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(input_scaled)[0]
                        classes = model.classes_
                        
                        proba_df = pd.DataFrame({
                            'Class': classes,
                            'Probability': proba
                        })
                        
                        fig = px.bar(
                            proba_df, x='Class', y='Probability',
                            title='Prediction Probabilities',
                            color='Probability'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
        
        elif input_method == "Upload File":
            st.write("Upload a file with the same feature columns.")
            uploaded_file = st.file_uploader("Upload CSV or Excel with prediction data", type=['csv', 'xlsx', 'xls'])
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        pred_df = pd.read_csv(uploaded_file)
                    else:
                        pred_df = pd.read_excel(uploaded_file)
                    
                    # Check if all features are present
                    missing_features = [f for f in features if f not in pred_df.columns]
                    
                    if missing_features:
                        st.error(f"Missing required features: {', '.join(missing_features)}")
                    else:
                        if st.button("Generate Predictions"):
                            try:
                                # Preprocess if needed
                                if preprocessor is not None:
                                    input_processed = preprocessor.transform(pred_df[features])
                                else:
                                    input_processed = pred_df[features]
                                
                                # Scale the input
                                input_scaled = scaler.transform(input_processed)
                                
                                # Make predictions
                                predictions = model.predict(input_scaled)
                                
                                # Add predictions to the dataframe
                                result_df = pred_df.copy()
                                result_df['Prediction'] = predictions
                                
                                st.write("### Prediction Results")
                                st.dataframe(result_df, use_container_width=True)
                                
                                # Download predictions
                                csv = result_df.to_csv(index=False)
                                b64 = base64.b64encode(csv.encode()).decode()
                                href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions (CSV)</a>'
                                st.markdown(href, unsafe_allow_html=True)
                                
                                # Visualize predictions
                                st.write("### Prediction Visualization")
                                
                                if model_info['task'] == "Classification":
                                    # Distribution of predicted classes
                                    pred_dist = pd.DataFrame(predictions).value_counts().reset_index()
                                    pred_dist.columns = ['Predicted Class', 'Count']
                                    
                                    fig = px.pie(
                                        pred_dist, values='Count', names='Predicted Class',
                                        title='Distribution of Predicted Classes'
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                else:  # Regression
                                    fig = px.histogram(
                                        result_df, x='Prediction',
                                        title='Distribution of Predictions',
                                        nbins=20
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            except Exception as e:
                                st.error(f"Error generating predictions: {str(e)}")
                
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)

def display_about():
    st.markdown('<div class="sub-header">About This Dashboard</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Advanced Data Analysis Dashboard
        
        This dashboard is designed to provide comprehensive data analysis capabilities, including:
        
        - **Data Exploration**: Upload, clean, and explore your datasets
        - **Visualization**: Generate interactive charts and graphs
        - **Machine Learning**: Train models and make predictions
        - **Export**: Save results and visualizations
        
        The application is built with Streamlit and integrates multiple data science libraries such as Pandas, Scikit-learn, Plotly, and more.
        
        ### Usage Guidelines
        
        1. Start by uploading a dataset in the "Data Upload & Exploration" section
        2. Explore your data and perform any necessary cleaning
        3. Create visualizations to understand patterns and relationships
        4. Build machine learning models to make predictions
        
        ### Technologies Used
        
        - **Streamlit**: Web application framework
        - **Pandas**: Data manipulation and analysis
        - **Scikit-learn**: Machine learning algorithms
        - **Plotly & Matplotlib**: Data visualization
        - **Seaborn**: Statistical visualizations
        - **NumPy**: Numerical computing
        """)
    
    with col2:
        st.image("https://www.svgrepo.com/show/416716/analytics-business-chart.svg", width=200)
        
        st.markdown("""
        ### Version Information
        
        **Version**: 2.1.0
        
        **Last Updated**: May 2025
        
        **Author**: Data Science Team
        
        **License**: MIT
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Feature Roadmap</div>', unsafe_allow_html=True)
    
    roadmap_col1, roadmap_col2 = st.columns(2)
    
    with roadmap_col1:
        st.markdown("""
        ### Coming Soon
        
        - Advanced time series forecasting
        - Natural language processing capabilities
        - Real-time data streaming integration
        - Enhanced model deployment options
        - Automated reporting features
        """)
    
    with roadmap_col2:
        st.markdown("""
        ### Under Consideration
        
        - Multi-user collaboration features
        - Deep learning model integration
        - Automated insights generation
        - Integration with data warehouses
        - Custom algorithm development
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Feedback section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Feedback</div>', unsafe_allow_html=True)
    
    feedback = st.text_area("Your feedback helps us improve. Let us know what you think:", height=100)
    
    feedback_type = st.radio("Feedback Type", ["General", "Bug Report", "Feature Request", "Question"])
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        email = st.text_input("Email (optional)")
    
    with col2:
        severity = st.select_slider("Priority", options=["Low", "Medium", "High"])
    
    with col3:
        if st.button("Submit Feedback"):
            if feedback:
                st.success("Thank you for your feedback! We'll review it soon.")
            else:
                st.error("Please enter your feedback before submitting.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Helper functions
def make_subplots(rows, cols, subplot_titles, vertical_spacing=0.1):
    """Helper function to create subplot figure"""
    fig = go.Figure()
    
    # Set the layout grid
    fig.update_layout(
        grid=dict(rows=rows, columns=cols),
        title_text=subplot_titles,
        height=rows * 300,
        margin=dict(t=50, l=50, r=50, b=50),
        hovermode="closest"
    )
    
    return fig

# Main function to run the app
if __name__ == "__main__":
    main() = pd.DataFrame(data.data, columns=data.feature_names)
                df['target'] = data.target
                st.session_state.data = df
                st.success("✅ Iris dataset loaded successfully!")
            
            elif sample_dataset == "Boston Housing":
                from sklearn.datasets import fetch_california_housing
                data = fetch_california_housing()
                df = pd.DataFrame(data.data, columns=data.feature_names)
                df['target'] = data.target
                st.session_state.data = df
                st.success("✅ California Housing dataset loaded successfully!")
            
            elif sample_dataset == "Wine Quality":
                from sklearn.datasets import load_wine
                data = load_wine()
                df = pd.DataFrame(data.data, columns=data.feature_names)
                df['target'] = data.target
                st.session_state.data = df
                st.success("✅ Wine Quality dataset loaded successfully!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Dashboard metrics
    st.markdown('<div class="sub-header">Dashboard Overview</div>', unsafe_allow_html=True)
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        if st.session_state.data is not None:
            st.metric("Loaded Dataset", f"{st.session_state.data.shape[0]} rows")
        else:
            st.metric("Loaded Dataset", "None")
    
    with metric_col2:
        if 'models' in st.session_state:
            st.metric("ML Models", len(st.session_state.models))
        else:
            st.metric("ML Models", 0)
    
    with metric_col3:
        # Simulated metric
        st.metric("Processing Speed", "98.2%", delta="3.4%")
    
    with metric_col4:
        # Simulated metric
        st.metric("System Status", "Online", delta="Active")

def display_data_upload():
    st.markdown('<div class="sub-header">Data Upload & Exploration</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        upload_tab, url_tab = st.tabs(["Upload File", "Import from URL"])
        
        with upload_tab:
            uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx', 'xls'])
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    st.session_state.data = df
                    st.success(f"✅ File uploaded successfully! Detected {df.shape[0]} rows and {df.shape[1]} columns.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        with url_tab:
            url = st.text_input("Enter URL to CSV or Excel file")
            if url:
                if st.button("Import Data"):
                    try:
                        with st.spinner("Downloading data..."):
                            if url.endswith('.csv'):
                                df = pd.read_csv(url)
                            else:
                                df = pd.read_excel(url)
                        
                        st.session_state.data = df
                        st.success(f"✅ Data imported successfully! Detected {df.shape[0]} rows and {df.shape[1]} columns.")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Data Options</div>', unsafe_allow_html=True)
        
        if st.session_state.data is not None:
            st.write("Data Processing:")
            if st.checkbox("Remove duplicate rows"):
                st.session_state.data = st.session_state.data.drop_duplicates()
                st.info(f"Duplicates removed. New shape: {st.session_state.data.shape}")
            
            if st.checkbox("Handle missing values"):
                missing_strategy = st.selectbox(
                    "Choose strategy",
                    ["Drop rows", "Fill with mean", "Fill with median", "Fill with zero"]
                )
                
                if missing_strategy == "Drop rows":
                    st.session_state.data = st.session_state.data.dropna()
                    st.info(f"Rows with missing values dropped. New shape: {st.session_state.data.shape}")
                elif missing_strategy == "Fill with mean":
                    st.session_state.data = st.session_state.data.fillna(st.session_state.data.mean(numeric_only=True))
                    st.info("Missing values filled with column means.")
                elif missing_strategy == "Fill with median":
                    st.session_state.data = st.session_state.data.fillna(st.session_state.data.median(numeric_only=True))
                    st.info("Missing values filled with column medians.")
                elif missing_strategy == "Fill with zero":
                    st.session_state.data = st.session_state.data.fillna(0)
                    st.info("Missing values filled with zeros.")
            
            st.write("Download processed data:")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Download CSV"):
                    csv = st.session_state.data.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="processed_data.csv">Download CSV File</a>'
                    st.markdown(href, unsafe_allow_html=True)
            
            with col2:
                if st.button("Download Excel"):
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        st.session_state.data.to_excel(writer, index=False)
                    b64 = base64.b64encode(output.getvalue()).decode()
                    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="processed_data.xlsx">Download Excel File</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Data exploration section
    if st.session_state.data is not None:
        df = st.session_state.data
        
        st.markdown('<div class="sub-header">Data Preview</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        preview_tab, stats_tab, missing_tab = st.tabs(["Preview", "Statistics", "Missing Values"])
        
        with preview_tab:
            rows_to_show = st.slider("Number of rows to display", 5, 100, 10)
            st.dataframe(df.head(rows_to_show), use_container_width=True)
            
            # Column information
            st.markdown("### Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes,
                'Non-Null Count': df.notnull().sum(),
                'Null Count': df.isnull().sum(),
                'Unique Values': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(col_info, use_container_width=True)
        
        with stats_tab:
            st.write("### Descriptive Statistics")
            st.dataframe(df.describe(include='all').transpose(), use_container_width=True)
            
            st.write("### Correlation Matrix")
            numeric_df = df.select_dtypes(include=['float64', 'int64'])
            if not numeric_df.empty:
                fig, ax = plt.subplots(figsize=(10, 8))
                corr = numeric_df.corr()
                sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
                st.pyplot(fig)
            else:
                st.info("No numerical columns found for correlation analysis.")
        
        with missing_tab:
            st.write("### Missing Values Analysis")
            
            missing_data = df.isnull().sum().reset_index()
            missing_data.columns = ['Column', 'Missing Count']
            missing_data['Missing Percentage'] = (missing_data['Missing Count'] / len(df)) * 100
            
            if missing_data['Missing Count'].sum() > 0:
                fig = px.bar(
                    missing_data,
                    x='Column',
                    y='Missing Percentage',
                    title='Missing Values by Column (%)',
                    text_auto='.2f',
                    color='Missing Percentage',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No missing values found in the dataset.")
        
        st.markdown('</div>', unsafe_allow_html=True)

def display_data_visualization():
    st.markdown('<div class="sub-header">Data Visualization</div>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("Please upload data first in the 'Data Upload & Exploration' section.")
        return
    
    df = st.session_state.data
    
    # Visualization options
    st.markdown('<div class="card">', unsafe_allow_html=True)
    viz_type = st.selectbox(
        "Select Visualization Type",
        ["Distribution Plots", "Relationship Plots", "Categorical Plots", "Time Series", "Interactive Plots", "3D Visualization"]
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.write("### Chart Options")
        
        # Get numerical and categorical columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        # Color theme
        color_theme = st.selectbox(
            "Color Theme",
            ["viridis", "plasma", "inferno", "magma", "cividis", "turbo", "blues", "reds", "greens"]
        )
        
        # Export options
        if st.button("Export Visualization"):
            st.session_state.export_viz = True
        else:
            st.session_state.export_viz = False
    
    with col1:
        if viz_type == "Distribution Plots":
            st.write("### Distribution Analysis")
            
            dist_plot_type = st.selectbox(
                "Plot Type", 
                ["Histogram", "KDE Plot", "Box Plot", "Violin Plot"]
            )
            
            if len(numeric_cols) > 0:
                selected_col = st.selectbox("Select Column", numeric_cols)
                
                if dist_plot_type == "Histogram":
                    fig = px.histogram(
                        df, x=selected_col, 
                        marginal="box", 
                        color_discrete_sequence=[color_theme],
                        title=f"Histogram of {selected_col}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif dist_plot_type == "KDE Plot":
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.kdeplot(data=df, x=selected_col, fill=True, ax=ax, palette=color_theme)
                    st.pyplot(fig)
                
                elif dist_plot_type == "Box Plot":
                    fig = px.box(
                        df, y=selected_col, 
                        title=f"Box Plot of {selected_col}", 
                        color_discrete_sequence=[color_theme]
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif dist_plot_type == "Violin Plot":
                    fig = px.violin(
                        df, y=selected_col, 
                        box=True, points="all", 
                        title=f"Violin Plot of {selected_col}",
                        color_discrete_sequence=[color_theme]
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numerical columns found for distribution analysis.")
        
        elif viz_type == "Relationship Plots":
            st.write("### Relationship Analysis")
            
            if len(numeric_cols) >= 2:
                rel_plot_type = st.selectbox(
                    "Plot Type", 
                    ["Scatter Plot", "Line Plot", "Bubble Plot", "Heatmap"]
                )
                
                x_col = st.selectbox("X-axis", numeric_cols)
                y_col = st.selectbox("Y-axis", [col for col in numeric_cols if col != x_col][:1] + [col for col in numeric_cols if col != x_col][1:])
                
                if rel_plot_type == "Scatter Plot":
                    color_col = st.selectbox("Color by (optional)", ["None"] + df.columns.tolist())
                    
                    if color_col != "None":
                        fig = px.scatter(
                            df, x=x_col, y=y_col, color=color_col,
                            title=f"{y_col} vs {x_col} (colored by {color_col})",
                            color_continuous_scale=color_theme if color_col in numeric_cols else None
                        )
                    else:
                        fig = px.scatter(
                            df, x=x_col, y=y_col,
                            title=f"{y_col} vs {x_col}",
                            color_discrete_sequence=[color_theme]
                        )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif rel_plot_type == "Line Plot":
                    fig = px.line(
                        df, x=x_col, y=y_col,
                        title=f"{y_col} vs {x_col}",
                        color_discrete_sequence=[color_theme]
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif rel_plot_type == "Bubble Plot":
                    size_col = st.selectbox("Size by", numeric_cols)
                    color_col = st.selectbox("Color by (optional)", ["None"] + df.columns.tolist())
                    
                    if color_col != "None":
                        fig = px.scatter(
                            df, x=x_col, y=y_col, size=size_col, color=color_col,
                            title=f"{y_col} vs {x_col} (size by {size_col}, colored by {color_col})",
                            color_continuous_scale=color_theme if color_col in numeric_cols else None
                        )
                    else:
                        fig = px.scatter(
                            df, x=x_col, y=y_col, size=size_col,
                            title=f"{y_col} vs {x_col} (size by {size_col})",
                            color_discrete_sequence=[color_theme]
                        )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif rel_plot_type == "Heatmap":
                    # Generate correlation matrix
                    corr_method = st.radio("Correlation Method", ["pearson", "spearman", "kendall"])
                    corr_matrix = df[numeric_cols].corr(method=corr_method)
                    
                    fig = px.imshow(
                        corr_matrix,
                        color_continuous_scale=color_theme,
                        title=f"Correlation Matrix ({corr_method})"
                    )
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("At least two numerical columns are required for relationship analysis.")
        
        elif viz_type == "Categorical Plots":
            st.write("### Categorical Analysis")
            
            if len(categorical_cols) > 0:
                cat_plot_type = st.selectbox(
                    "Plot Type", 
                    ["Bar Chart", "Pie Chart", "Count Plot", "Grouped Bar Chart"]
                )
                
                cat_col = st.selectbox("Category Column", categorical_cols)
                
                if cat_plot_type == "Bar Chart":
                    value_col = st.selectbox("Value Column", numeric_cols if numeric_cols else ["Count"])
                    
                    if value_col == "Count":
                        fig = px.bar(
                            df[cat_col].value_counts().reset_index(),
                            x="index", y=cat_col,
                            title=f"Count of {cat_col}",
                            color="index",
                            color_discrete_sequence=px.colors.sequential.get(color_theme, px.colors.sequential.Viridis)
                        )
                        fig.update_layout(xaxis_title=cat_col, yaxis_title="Count")
                    else:
                        fig = px.bar(
                            df.groupby(cat_col)[value_col].mean().reset_index(),
                            x=cat_col, y=value_col,
                            title=f"Average {value_col} by {cat_col}",
                            color=cat_col,
                            color_discrete_sequence=px.colors.sequential.get(color_theme, px.colors.sequential.Viridis)
                        )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif cat_plot_type == "Pie Chart":
                    fig = px.pie(
                        df[cat_col].value_counts().reset_index(),
                        values=cat_col, names="index",
                        title=f"Distribution of {cat_col}",
                        color_discrete_sequence=px.colors.sequential.get(color_theme, px.colors.sequential.Viridis)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif cat_plot_type == "Count Plot":
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.countplot(data=df, x=cat_col, palette=color_theme, ax=ax)
                    plt.xticks(rotation=45, ha="right")
                    plt.tight_layout()
                    st.pyplot(fig)
                
                elif cat_plot_type == "Grouped Bar Chart":
                    if len(categorical_cols) >= 2:
                        group_col = st.selectbox("Group By", [col for col in categorical_cols if col != cat_col])
                        value_col = st.selectbox("Value Column", numeric_cols if numeric_cols else ["Count"])
                        
                        if value_col == "Count":
                            count_df = df.groupby([cat_col, group_col]).size().reset_index(name="Count")
                            fig = px.bar(
                                count_df,
                                x=cat_col, y="Count", color=group_col,
                                title=f"Count by {cat_col} and {group_col}",
                                barmode="group",
                                color_discrete_sequence=px.colors.sequential.get(color_theme, px.colors.sequential.Viridis)
                            )
                        else:
                            agg_df = df.groupby([cat_col, group_col])[value_col].mean().reset_index()
                            fig = px.bar(
                                agg_df,
                                x=cat_col, y=value_col, color=group_col,
                                title=f"Average {value_col} by {cat_col} and {group_col}",
                                barmode="group",
                                color_discrete_sequence=px.colors.sequential.get(color_theme, px.colors.sequential.Viridis)
                            )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("At least two categorical columns are required for grouped bar chart.")
            else:
                st.info("No categorical columns found for categorical analysis.")
        
        elif viz_type == "Time Series":
            st.write("### Time Series Analysis")
            
            # Check for potential date columns
            date_cols = []
            for col in df.columns:
                if df[col].dtype == 'datetime64[ns]':
                    date_cols.append(col)
                elif df[col].dtype == 'object':
                    try:
                        pd.to_datetime(df[col])
                        date_cols.append(col)
                    except:
                        pass
            
            if date_cols:
                date_col = st.selectbox("Select Date Column", date_cols)
                
                # Convert to datetime if not already
                if df[date_col].dtype != 'datetime64[ns]':
                    try:
                        df[date_col] = pd.to_datetime(df[date_col])
                    except Exception as e:
                        st.error(f"Error converting {date_col} to datetime: {str(e)}")
                        return
                
                ts_plot_type = st.selectbox(
                    "Plot Type", 
                    ["Line Plot", "Area Plot", "Bar Plot", "Decomposition"]
                )
                
                value_col = st.selectbox("Value Column", numeric_cols)
                
                if ts_plot_type == "Line Plot":
                    group_col = st.selectbox("Group By (optional)", ["None"] + categorical_cols)
                    
                    if group_col != "None":
                        fig = px.line(
                            df