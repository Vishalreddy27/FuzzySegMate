import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import skfuzzy as fuzz
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

# Set page config with wider layout
st.set_page_config(
    page_title="Fuzzy C-Means Customer Segmentation",
    page_icon="🧑‍💼",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Add custom CSS for better appearance and text visibility
st.markdown("""
<style>
    /* Improve visibility on dark background */
    .main .block-container {padding-top: 2rem;}
    h1 {color: #FFFFFF; font-size: 2.5rem !important; margin-bottom: 1.5rem; text-shadow: 0 1px 2px rgba(0,0,0,0.1);}
    h2 {color: #FFFFFF; font-size: 2rem !important; margin-top: 2rem; margin-bottom: 1rem;}
    h3 {color: #FFFFFF; font-size: 1.7rem !important; margin-top: 1.5rem; margin-bottom: 1rem;}
    h4 {color: #FFFFFF; font-size: 1.4rem !important; margin-bottom: 0.8rem;}
    p, li, label, div.stMarkdown {color: #FFFFFF !important; font-size: 1.05rem !important; line-height: 1.6 !important;}
    
    /* Improve caption visibility */
    .caption, small {color: #FFFFFF !important; font-size: 0.9rem !important;}
    
    /* Make form elements more visible */
    div[data-testid="stForm"] {background-color: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 0.75rem; border: 1px solid rgba(255,255,255,0.2);}
    
    /* Better section headings */
    .section-header {
        background-color: #4B5563;
        color: white;
        padding: 12px 15px;
        border-radius: 8px;
        font-weight: 600;
        margin: 20px 0 15px 0;
        border-left: 4px solid #60A5FA;
    }
    
    /* Improve form section visibility */
    .form-section {
        background-color: rgba(255,255,255,0.05);
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
        border-left: 4px solid #60A5FA;
    }
    .form-section h4 {
        color: #FFFFFF;
        margin-top: 0;
        font-weight: 600;
    }
    
    /* Enhance explanation box */
    .explanation-box {
        background-color: rgba(59, 130, 246, 0.2);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 30px;
        border-left: 5px solid #3B82F6;
    }
    .explanation-box p {color: #FFFFFF !important;}
    .explanation-box h3 {color: #FFFFFF !important; margin-top: 0; margin-bottom: 15px;}
    
    /* Result boxes with better visibility */
    .result-box {
        background-color: rgba(255,255,255,0.1);
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        border-left: 5px solid #60A5FA;
    }
    
    /* Cluster meanings box */
    .cluster-meanings-box {
        background-color: rgba(79, 70, 229, 0.15);
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        border: 1px solid rgba(79, 70, 229, 0.3);
    }
    
    /* Individual cluster card */
    .cluster-card {
        background-color: rgba(255,255,255,0.1);
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        border-left: 4px solid;
    }
    
    /* Improve expander styling */
    .stExpander {border: 1px solid rgba(255,255,255,0.2); border-radius: 0.5rem; margin-bottom: 1rem;}
    
    /* Chart improvements */
    .stPlotlyChart, .stPyplot {
        background-color: rgba(255,255,255,0.05);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Feature list styling */
    .feature-item {
        padding: 8px 15px;
        background-color: rgba(255,255,255,0.05);
        border-radius: 5px;
        margin-bottom: 8px;
        border-left: 3px solid #60A5FA;
    }
    
    /* Make buttons more visible */
    button[kind="primaryFormSubmit"] {
        background-color: #3B82F6 !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.5rem !important;
        font-size: 1.1rem !important;
        border-radius: 0.5rem !important;
        border: none !important;
    }
    
    /* Divider for visual separation */
    .divider {
        height: 1px;
        background-color: rgba(255,255,255,0.2);
        margin: 25px 0;
    }
    
    /* Better info box */
    div.stAlert {background-color: rgba(59, 130, 246, 0.2) !important; color: white !important;}
    div.stAlert p {color: white !important;}
</style>
""", unsafe_allow_html=True)

# Load the pre-trained FCM model
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_model():
    try:
        with open('fcm_model.pkl', 'rb') as file:
            model = pickle.load(file)
            # Override n_clusters to 4 if needed
            if 'n_clusters' in model and model['n_clusters'] != 4:
                st.warning("The existing model uses a different number of clusters. Retraining with 4 clusters...")
                # Need to retrain the model with 4 clusters
                dataset_result, _ = load_dataset()
                if dataset_result is not None:
                    model = train_fcm_model(dataset_result, n_clusters=4)
            return model, None
    except Exception as e:
        # If model doesn't exist or can't be loaded, try to create a new one
        st.warning(f"Failed to load model: {str(e)}. Attempting to create a new model...")
        try:
            dataset_result, _ = load_dataset()
            if dataset_result is not None:
                model = train_fcm_model(dataset_result, n_clusters=4)
                return model, None
            else:
                return None, "Failed to load dataset for model creation"
        except Exception as train_error:
            return None, f"Failed to create model: {str(train_error)}"

# Function to train a new FCM model with specified number of clusters
def train_fcm_model(dataset, n_clusters=4):
    # Features for clustering - using numerical features only
    features = ['Age', 'Total Spend', 'Items Purchased', 'Average Rating', 'Days Since Last Purchase']
    
    # Extract features from dataset
    if all(feature in dataset.columns for feature in features):
        X = dataset[features].values
    else:
        # Use whatever numerical columns are available
        numerical_cols = dataset.select_dtypes(include=['int64', 'float64']).columns.tolist()
        features = numerical_cols
        X = dataset[features].values
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Transpose data for skfuzzy (features as rows, samples as columns)
    X_scaled_transposed = X_scaled.T
    
    # Apply Fuzzy C-Means
    fpcs = []
    centers_list = []
    
    # Use fuzziness parameter of 2.0 (standard for FCM)
    m = 2.0
    
    # Train the model
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        X_scaled_transposed, n_clusters, m, error=0.005, maxiter=1000, init=None
    )
    
    # Get cluster labels
    cluster_labels = np.argmax(u, axis=0)
    
    # Add membership degrees to the dataset
    for i in range(n_clusters):
        dataset[f'Membership_Cluster_{i}'] = u[i]
    
    # Add cluster labels to the dataset
    dataset['Cluster'] = cluster_labels
    
    # Create the model dictionary
    model = {
        'centers': cntr,
        'u': u,
        'features': features,
        'scaler': scaler,
        'n_clusters': n_clusters,
        'm': m,
        'cluster_labels': cluster_labels,
        'fpc': fpc
    }
    
    # Save the model
    try:
        with open('fcm_model.pkl', 'wb') as file:
            pickle.dump(model, file)
        st.success(f"Successfully created and saved a new FCM model with {n_clusters} clusters!")
    except Exception as e:
        st.error(f"Failed to save model: {str(e)}")
    
    return model

# Load the dataset to get feature ranges and for visualization
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_dataset():
    try:
        df = pd.read_csv('E-commerce Customer Behavior - Sheet1.csv')
        return df, None
    except Exception as e:
        return None, str(e)

# Function to preprocess user input
def preprocess_input(user_input, model):
    # Extract features and scaler from the model
    features = model['features']
    scaler = model['scaler']
    
    # Create a DataFrame with the user input
    user_data = pd.DataFrame([user_input])
    
    # Extract just the numerical features that were used for clustering
    user_numerical = {}
    for feature in features:
        if feature in user_data.columns:
            # Use the provided value
            user_numerical[feature] = user_data[feature].values[0]
        else:
            # If a feature is missing, assign a default value (this shouldn't happen in the UI)
            user_numerical[feature] = 0
    
    # Convert to DataFrame
    user_df = pd.DataFrame([user_numerical])
    
    # Scale the data using the model's scaler
    user_scaled = scaler.transform(user_df)
    
    return user_scaled, features

# Function to predict the cluster and membership degrees
def predict_fuzzy_cluster(user_input, model):
    # Preprocess the user input
    user_scaled, features = preprocess_input(user_input, model)
    
    # Transpose for skfuzzy's format (features as rows, samples as columns)
    user_scaled_transposed = user_scaled.T
    
    # Get the model parameters
    centers = model['centers']
    m = model['m']  # fuzziness parameter
    
    # Calculate the membership degrees for the user input
    # Using Fuzzy C-Means prediction function
    u, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
        user_scaled_transposed, centers, m, error=0.005, maxiter=1000
    )
    
    # Get the primary cluster (highest membership)
    primary_cluster = np.argmax(u, axis=0)[0]
    
    # Get the membership degrees for each cluster
    membership_degrees = u[:, 0]
    
    return primary_cluster, membership_degrees

# Get cluster descriptions based on the dataset and FCM model
def get_cluster_descriptions(dataset, model):
    # Add the cluster labels if not already present
    if 'Cluster' not in dataset.columns:
        dataset['Cluster'] = model['cluster_labels']
    
    # Get the features that were used for clustering
    features = model['features']
    
    # Calculate cluster means
    cluster_means = dataset.groupby('Cluster')[features].mean()
    
    # Calculate overall means
    overall_means = dataset[features].mean()
    
    # Create descriptions for each cluster
    descriptions = {}
    
    # Cluster colors for better visualization
    cluster_colors = ['#4285F4', '#EA4335', '#34A853', '#FBBC05', '#FF6D01', '#46BDC6']
    
    # Predefined cluster meanings based on common customer segments
    cluster_meanings = {
        0: {
            "name": "High-Value Loyal Customers",
            "description": "Customers who spend more, purchase frequently, and give high ratings",
            "marketing_strategy": "Reward loyalty with exclusive offers and early access to new products"
        },
        1: {
            "name": "Price-Sensitive Occasional Shoppers",
            "description": "Customers with average spending but longer periods between purchases",
            "marketing_strategy": "Send targeted discounts to encourage more frequent purchases"
        },
        2: {
            "name": "New Potential High-Value Customers",
            "description": "Recent customers with high initial spend but limited history",
            "marketing_strategy": "Focus on building relationships and encouraging repeat purchases"
        },
        3: {
            "name": "At-Risk Customers",
            "description": "Previously active customers who haven't purchased recently",
            "marketing_strategy": "Re-engagement campaigns with personalized offers based on past purchases"
        }
    }
    
    for i, cluster in enumerate(sorted(dataset['Cluster'].unique())):
        cluster_data = cluster_means.loc[cluster]
        
        # Identify high and low value features
        high_features = [f for f in features if cluster_data[f] > overall_means[f]]
        low_features = [f for f in features if cluster_data[f] < overall_means[f]]
        
        # Use predefined meanings if available, otherwise generate based on features
        if int(cluster) < len(cluster_meanings):
            segment_name = cluster_meanings[int(cluster)]["name"]
            description = cluster_meanings[int(cluster)]["description"]
            marketing_strategy = cluster_meanings[int(cluster)]["marketing_strategy"]
        else:
            # Create a description based on feature values
            if 'Total Spend' in high_features and 'Items Purchased' in high_features:
                if 'Average Rating' in high_features:
                    segment_name = "High-Value Loyal Customers"
                    description = "Customers who spend more, purchase frequently, and give high ratings"
                    marketing_strategy = "Reward loyalty with exclusive offers and early access to new products"
                else:
                    segment_name = "Big Spenders"
                    description = "Customers who spend a lot but may not be fully satisfied"
                    marketing_strategy = "Focus on improving customer experience and satisfaction"
            elif 'Total Spend' in low_features and 'Items Purchased' in low_features:
                if 'Days Since Last Purchase' in high_features:
                    segment_name = "Lapsed Customers"
                    description = "Customers who haven't purchased in a while and spend less"
                    marketing_strategy = "Re-engagement campaigns with special offers"
                else:
                    segment_name = "Low-Value Customers"
                    description = "Customers who spend less and purchase less frequently"
                    marketing_strategy = "Low-cost engagement strategies and occasional offers"
            else:
                segment_name = f"Segment {cluster}"
                description = "Customers with mixed purchasing patterns"
                marketing_strategy = "Test different marketing approaches to find what resonates"
        
        # Assign a color to this cluster (cycling through the color list)
        color = cluster_colors[i % len(cluster_colors)]
        
        descriptions[cluster] = {
            "name": segment_name,
            "description": description,
            "high_features": high_features,
            "low_features": low_features,
            "color": color,
            "marketing_strategy": marketing_strategy
        }
    
    return descriptions

# Function to create a radar chart for visualizing cluster memberships
def create_membership_radar_chart(membership_degrees, n_clusters):
    # Set up the figure
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, polar=True)
    
    # Set up angles for each cluster
    angles = np.linspace(0, 2*np.pi, n_clusters, endpoint=False).tolist()
    
    # Make the plot circular by adding the first value at the end
    angles.append(angles[0])
    membership_values = membership_degrees.tolist()
    membership_values.append(membership_values[0])
    
    # Plot the membership values with better styling
    ax.plot(angles, membership_values, linewidth=3, linestyle='solid', color='#4285F4')
    ax.fill(angles, membership_values, alpha=0.4, color='#4285F4')
    
    # Set labels for each angle
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f'Cluster {i}' for i in range(n_clusters)], fontsize=11)
    
    # Set y-ticks
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set title
    plt.title('Membership Degrees Visualization', size=16, pad=20, fontweight='bold')
    
    # Add some styling
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    return fig

# Create a human-readable interpretation of features
def get_feature_friendly_name(feature):
    feature_map = {
        'Age': 'Age',
        'Total Spend': 'Total Spending Amount',
        'Items Purchased': 'Number of Items Purchased',
        'Average Rating': 'Average Product Rating',
        'Days Since Last Purchase': 'Days Since Last Purchase'
    }
    return feature_map.get(feature, feature)

# Display all cluster meanings in a dedicated section
def display_cluster_meanings(cluster_descriptions):
    st.markdown('<h3 style="color:#FFFFFF;">Customer Segment Meanings</h3>', unsafe_allow_html=True)
    
    # Display each cluster with its meaning in card format
    for cluster, info in sorted(cluster_descriptions.items()):
        st.markdown(f"""
        <div class="cluster-card" style="border-left-color:{info['color']}">
            <h4 style="color:#FFFFFF;margin-top:0;">{info['name']} (Segment {cluster})</h4>
            <p style="font-size:16px;color:#FFFFFF;"><b>Description:</b> {info['description']}</p>
            <p style="font-size:16px;color:#FFFFFF;"><b>Marketing Strategy:</b> {info['marketing_strategy']}</p>
        </div>
        """, unsafe_allow_html=True)

# Main application
def main():
    # Load the model and dataset
    model_result, model_error = load_model()
    dataset_result, dataset_error = load_dataset()
    
    # Check if dataset loaded successfully
    if dataset_result is None:
        st.error(f"Failed to load dataset: {dataset_error}")
        st.stop()
    else:
        dataset = dataset_result
    
    # Check if model loaded successfully
    if model_result is None:
        st.error(f"Failed to load Fuzzy C-Means model: {model_error}")
        st.stop()
    else:
        fcm_model = model_result
    
    # Ensure we're using 4 clusters
    if fcm_model['n_clusters'] != 4:
        st.warning("Model still doesn't have 4 clusters. Attempting to fix...")
        # Force update to 4 clusters in the running model
        fcm_model['n_clusters'] = 4
    
    # Extract model parameters
    n_clusters = fcm_model['n_clusters']
    
    # Get cluster descriptions
    cluster_descriptions = get_cluster_descriptions(dataset, fcm_model)
    
    # Application title and description with improved styling
    st.title("📊 Fuzzy Customer Segmentation Analysis")
    
    # Display the number of clusters being used
    st.markdown(f"""
    <div style="background-color:rgba(59, 130, 246, 0.1);padding:10px;border-radius:10px;margin-bottom:15px;">
        <p style="margin:0;color:#FFFFFF;"><strong>Current model settings:</strong> Using {n_clusters} customer segments</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Improved explanation box with better visibility
    st.markdown("""
    <div class="explanation-box">
    <h3>What is Fuzzy Customer Segmentation?</h3>
    <p>Unlike traditional segmentation that places customers into single, distinct groups, 
    <b>fuzzy segmentation</b> recognizes that customers often share characteristics with <b>multiple segments</b>.</p>
    <p>Each customer receives a <b>membership score</b> for each segment, showing how strongly they 
    belong to different customer groups - a more realistic approach to understanding customer behavior.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display all cluster meanings at the top for better understanding
    st.markdown('<div class="cluster-meanings-box">', unsafe_allow_html=True)
    display_cluster_meanings(cluster_descriptions)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create two columns for the main layout
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.markdown('<div class="section-header">✏️ Enter Your Customer Details</div>', unsafe_allow_html=True)
        
        # Create input form with better styling
        with st.form(key="customer_form"):
            # Basic info section
            st.markdown('<div class="form-section"><h4>👤 Basic Information</h4></div>', unsafe_allow_html=True)
            gender = st.selectbox("Gender", options=["Male", "Female"])
            age = st.slider("Age", min_value=18, max_value=80, value=30)
            city = st.selectbox("City", options=["New York", "Los Angeles", "Chicago", "San Francisco", "Miami", "Houston"])
            membership = st.selectbox("Membership Type", options=["Bronze", "Silver", "Gold"])
            
            # Purchasing behavior section
            st.markdown('<div class="form-section"><h4>💰 Purchasing Behavior</h4></div>', unsafe_allow_html=True)
            total_spend = st.number_input("Total Spend ($)", min_value=0.0, max_value=2000.0, value=500.0, step=50.0)
            items_purchased = st.slider("Items Purchased", min_value=1, max_value=30, value=10)
            avg_rating = st.slider("Average Rating (1-5)", min_value=1.0, max_value=5.0, value=4.0, step=0.1)
            days_since_purchase = st.slider("Days Since Last Purchase", min_value=1, max_value=60, value=20)
            discount_applied = st.checkbox("Discount Applied on Last Purchase")
            
            # Customer satisfaction section
            st.markdown('<div class="form-section"><h4>😊 Customer Satisfaction</h4></div>', unsafe_allow_html=True)
            satisfaction = st.selectbox("Satisfaction Level", options=["Satisfied", "Neutral", "Unsatisfied"])
            
            # Submit button with better styling
            st.markdown('<div style="display: flex; justify-content: center; margin-top: 25px;">', unsafe_allow_html=True)
            submit_button = st.form_submit_button(label="🔍 Analyze My Customer Profile")
            st.markdown('</div>', unsafe_allow_html=True)
    
        # When form is submitted
        if submit_button:
            # Convert discount_applied to numerical (1 for True, 0 for False)
            discount_applied_num = 1 if discount_applied else 0
            
            # Create a dictionary of user inputs focusing on numerical values
            user_input = {
                'Age': age,
                'Total Spend': total_spend,
                'Items Purchased': items_purchased,
                'Average Rating': avg_rating,
                'Days Since Last Purchase': days_since_purchase
            }
            
            # Add a spinner while calculating
            with st.spinner("Analyzing your customer profile..."):
                # Predict the cluster and membership degrees
                primary_cluster, membership_degrees = predict_fuzzy_cluster(user_input, fcm_model)
            
            # Primary segment result with better visibility
            primary_color = cluster_descriptions[primary_cluster]['color']
            st.markdown(f"""
            <div class="result-box" style="border-left-color:{primary_color};">
                <h3 style="color:#FFFFFF;margin-top:0;">Your Primary Segment: {cluster_descriptions[primary_cluster]['name']}</h3>
                <p style="font-size:16px;color:#FFFFFF;"><b>Description:</b> {cluster_descriptions[primary_cluster]['description']}</p>
                <p style="font-size:16px;color:#FFFFFF;"><b>Recommended Strategy:</b> {cluster_descriptions[primary_cluster]['marketing_strategy']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Visual separator
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            
            # Show membership visualization with better heading
            st.markdown('<h3 style="color:#FFFFFF;">Your Segment Membership Profile</h3>', unsafe_allow_html=True)
            
            # Create a bar chart of membership degrees with better labels
            membership_df = pd.DataFrame({
                'Cluster': [f"{cluster_descriptions[cluster]['name']}" for cluster in range(n_clusters)],
                'Membership Degree': membership_degrees
            })
            
            # Sort by membership degree for better visualization
            membership_df = membership_df.sort_values('Membership Degree', ascending=False)
            
            # Show the bar chart
            st.bar_chart(membership_df.set_index('Cluster')['Membership Degree'])
            
            # Explanation of membership degrees with better visibility
            st.info("""
            **Understanding Membership Degrees:** 
            
            The values above (between 0 and 1) represent how strongly you belong to each segment. 
            A value closer to 1 indicates stronger membership in that segment.
            """)
            
            # Display a radar chart with better heading
            st.markdown('<h3 style="color:#FFFFFF;">Membership Visualization</h3>', unsafe_allow_html=True)
            radar_fig = create_membership_radar_chart(membership_degrees, n_clusters)
            st.pyplot(radar_fig)
            
            # Visual separator
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            
            # Show key characteristics of the primary cluster with better formatting
            st.markdown(f"""
            <h3 style="color:#FFFFFF;">Key Characteristics of Your Primary Segment</h3>
            <div class="result-box">
            """, unsafe_allow_html=True)
            
            high_features = cluster_descriptions[primary_cluster]['high_features']
            low_features = cluster_descriptions[primary_cluster]['low_features']
            
            if high_features:
                st.markdown(f"""<p style='color:#10B981;font-weight:600;font-size:16px;'>
                               ▲ Higher than average in:</p>""", unsafe_allow_html=True)
                for feature in high_features:
                    st.markdown(f"""<div class="feature-item">
                                • {get_feature_friendly_name(feature)}</div>""", unsafe_allow_html=True)
            
            if low_features:
                st.markdown(f"""<p style='color:#F59E0B;font-weight:600;font-size:16px;margin-top:15px;'>
                               ▼ Lower than average in:</p>""", unsafe_allow_html=True)
                for feature in low_features:
                    st.markdown(f"""<div class="feature-item">
                                • {get_feature_friendly_name(feature)}</div>""", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Provide a fuzzy interpretation with better heading
            st.markdown("<h3 style='color:#FFFFFF;'>Your Fuzzy Segment Analysis</h3>", unsafe_allow_html=True)
            
            # Find secondary cluster (second highest membership)
            membership_sorted = np.argsort(membership_degrees)[::-1]
            secondary_cluster = membership_sorted[1]
            secondary_membership = membership_degrees[secondary_cluster]
            
            # Only show if the secondary membership is significant
            if secondary_membership > 0.3:
                secondary_color = cluster_descriptions[secondary_cluster]['color']
                st.markdown(f"""
                <div class="result-box">
                <p style="font-size:16px;line-height:1.6;color:#FFFFFF;">
                While you primarily belong to the <b style="color:{primary_color};">
                {cluster_descriptions[primary_cluster]['name']}</b> segment 
                (<b>{membership_degrees[primary_cluster]:.2f}</b> membership degree), you also significantly share characteristics 
                with the <b style="color:{secondary_color};">
                {cluster_descriptions[secondary_cluster]['name']}</b> segment 
                (<b>{secondary_membership:.2f}</b> membership degree).
                </p>
                <p style="font-style:italic;margin-bottom:0;color:#FFFFFF;">This means marketing strategies targeting either segment may resonate with you.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box">
                <p style="font-size:16px;line-height:1.6;color:#FFFFFF;">
                You have a <b>strong affiliation</b> with the <b style="color:{primary_color};">
                {cluster_descriptions[primary_cluster]['name']}</b> segment 
                (membership degree: <b>{membership_degrees[primary_cluster]:.2f}</b>) and minimal overlap with other segments.
                </p>
                <p style="font-style:italic;margin-bottom:0;color:#FFFFFF;">This means you fit very clearly into this customer segment, and marketing strategies for this segment would be highly relevant to you.</p>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="section-header">📈 Customer Segments Insights</div>', unsafe_allow_html=True)
        
        # Show the membership distribution visualization with better visibility
        if os.path.exists('membership_distribution.png'):
            st.markdown('<h3 style="color:#FFFFFF;">Distribution of Membership Values</h3>', unsafe_allow_html=True)
            st.image('membership_distribution.png')
            st.markdown('<p style="color:#FFFFFF;">This chart shows how membership values are distributed across all customers in the dataset, demonstrating the "fuzzy" nature of the segmentation.</p>', unsafe_allow_html=True)
        
        # Visual separator
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Show the cluster characteristics visualization with better visibility
        if os.path.exists('cluster_characteristics.png'):
            st.markdown('<h3 style="color:#FFFFFF;">Cluster Characteristics Comparison</h3>', unsafe_allow_html=True)
            st.image('cluster_characteristics.png')
            st.markdown('<p style="color:#FFFFFF;">This chart shows the normalized feature values for each cluster, highlighting what makes each segment unique.</p>', unsafe_allow_html=True)
        
        # Visual separator
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Show segment characteristics using better styled expandable sections
        st.markdown('<h3 style="color:#FFFFFF;">All Customer Segments Details</h3>', unsafe_allow_html=True)
        
        for cluster in sorted(cluster_descriptions.keys()):
            info = cluster_descriptions[cluster]
            
            with st.expander(f"🔎 {info['name']} (Segment {cluster})"):
                st.markdown(f"""
                <div style="background-color:rgba(255,255,255,0.1);padding:15px;border-radius:8px;border-left:3px solid {info['color']};">
                <p style="font-size:16px;color:#FFFFFF;"><b>Description:</b> {info['description']}</p>
                <p style="font-size:16px;color:#FFFFFF;"><b>Marketing Strategy:</b> {info['marketing_strategy']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Higher than average characteristics with better visibility
                st.markdown(f"""<div style="background-color:rgba(16,185,129,0.1);padding:10px;border-radius:8px;border-left:3px solid #10B981;">
                <p style="color:#FFFFFF;font-weight:600;">Higher than average in:</p>
                </div>""", unsafe_allow_html=True)
                
                if info['high_features']:
                    for feature in info['high_features']:
                        st.markdown(f"""<div class="feature-item">• {get_feature_friendly_name(feature)}</div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""<div class="feature-item">• None</div>""", unsafe_allow_html=True)
                
                # Lower than average characteristics with better visibility
                st.markdown(f"""<div style="background-color:rgba(245,158,11,0.1);padding:10px;border-radius:8px;border-left:3px solid #F59E0B;margin-top:15px;">
                <p style="color:#FFFFFF;font-weight:600;">Lower than average in:</p>
                </div>""", unsafe_allow_html=True)
                
                if info['low_features']:
                    for feature in info['low_features']:
                        st.markdown(f"""<div class="feature-item">• {get_feature_friendly_name(feature)}</div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""<div class="feature-item">• None</div>""", unsafe_allow_html=True)
                
                # Average membership information with better visibility
                st.markdown("<br>", unsafe_allow_html=True)
                membership_column = f'Membership_Cluster_{cluster}'
                if membership_column in dataset.columns:
                    avg_membership = dataset[membership_column].mean()
                    st.markdown(f"""<div style="background-color:rgba(255,255,255,0.1);padding:15px;border-radius:8px;">
                    <p style="font-weight:600;color:#FFFFFF;margin-bottom:5px;">Average membership degree:</p>
                    <p style="color:#60A5FA;font-weight:600;font-size:1.2rem;margin:0;">{avg_membership:.4f}</p>
                    </div>""", unsafe_allow_html=True)
                else:
                    # If membership columns don't exist in the dataset
                    st.markdown(f"""<div style="background-color:rgba(255,255,255,0.1);padding:15px;border-radius:8px;">
                    <p style="font-weight:600;color:#FFFFFF;margin-bottom:5px;">Average membership degree:</p>
                    <p style="color:#60A5FA;font-size:1.2rem;margin:0;">Not available</p>
                    </div>""", unsafe_allow_html=True)
        
        # Visual separator
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Business recommendations section with improved visibility
        st.markdown("""
        <div style="padding:20px;background-color:rgba(16,185,129,0.1);border-radius:10px;border-left:4px solid #10B981;">
        <h3 style="color:#FFFFFF;margin-top:0;">Business Recommendations</h3>
        <p style="color:#FFFFFF;">Fuzzy segmentation provides more nuanced marketing opportunities:</p>
        <ul style="color:#FFFFFF;">
            <li style="margin-bottom:10px;"><b>Targeted Marketing:</b> Primary segments receive focused campaigns</li>
            <li style="margin-bottom:10px;"><b>Cross-Segment Strategies:</b> Secondary segments receive adapted messaging</li>
            <li style="margin-bottom:10px;"><b>Personalization:</b> Membership degrees guide the level of personalization</li>
            <li><b>Transition Tracking:</b> Monitor how customers move between segments over time</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main() 