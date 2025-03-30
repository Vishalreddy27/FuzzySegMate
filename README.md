# Fuzzy C-Means Customer Segmentation Application

This repository contains an interactive web application for customer segmentation in e-commerce using Fuzzy C-Means (FCM) clustering. Unlike traditional segmentation that places customers into single, distinct groups, fuzzy segmentation recognizes that customers often share characteristics with multiple segments.

## Features

- **Fuzzy Clustering**: Shows membership degrees across 4 distinct customer segments
- **Interactive UI**: Enter customer information and instantly see segment predictions
- **Visualization**: Bar charts and radar charts show membership distribution across segments
- **Detailed Analysis**: Provides business insights and marketing recommendations for each segment
- **Pre-trained Model**: Uses a trained FCM model with 4 customer segments

## Project Files

### Core Application Files

1. **`fcm_ui.py`** (Main Application)
   - The primary Streamlit web application
   - Provides an interactive interface for customer segmentation
   - Features:
     - Input form for customer details
     - Real-time segment prediction
     - Membership degree visualization
     - Detailed segment analysis
     - Marketing recommendations
   - Uses the pre-trained FCM model for predictions

2. **`fcm_model.pkl`** (Pre-trained Model)
   - Contains the trained Fuzzy C-Means clustering model
   - Includes:
     - Cluster centers
     - Membership matrix
     - Feature scaler
     - Model parameters
   - Configured with 4 customer segments
   - Used by the main application for predictions

3. **`train_fcm_model.py`** (Model Training Script)
   - Script for training the Fuzzy C-Means model
   - Features:
     - Data preprocessing
     - Feature scaling
     - FCM model training
     - Visualization generation
   - Creates:
     - New model file (`fcm_model.pkl`)
     - Membership distribution plot
     - Cluster characteristics visualization

4. **`check_model.py`** (Model Verification Tool)
   - Utility script to verify model configuration
   - Checks:
     - Number of clusters
     - Feature set
     - Model structure
     - Cluster distribution
   - Helps ensure model integrity and proper configuration

### Data and Analysis Files

5. **`E-commerce Customer Behavior - Sheet1.csv`** (Dataset)
   - Contains customer data used for model training
   - Features include:
     - Age
     - Total Spend
     - Items Purchased
     - Average Rating
     - Days Since Last Purchase
   - Used to train the FCM model

6. **`Pre_Understanding.ipynb`** (Data Analysis Notebook)
   - Jupyter notebook for exploratory data analysis
   - Contains:
     - Data loading and preprocessing
     - Feature analysis
     - Statistical summaries
     - Initial clustering attempts
     - Data visualization
   - Used for understanding the dataset before model development

### Visualization Files

7. **`membership_distribution.png`**
   - Shows the distribution of membership values across clusters
   - Helps understand the "fuzzy" nature of the segmentation
   - Visualizes how customers belong to multiple segments

8. **`cluster_characteristics.png`**
   - Heatmap showing normalized feature values for each cluster
   - Helps identify key characteristics of each segment
   - Visualizes cluster differences and similarities

### Configuration Files

9. **`requirements.txt`**
   - Lists all Python package dependencies
   - Includes specific versions for compatibility
   - Required packages:
     - streamlit==1.12.0
     - pandas==2.2.3
     - numpy==2.0.2
     - matplotlib==3.9.4
     - seaborn==0.13.2
     - scikit-learn==1.6.1
     - scikit-fuzzy==0.5.0
     - joblib==1.4.2
     - altair==4.2.2

## Installation

This application runs in a Python virtual environment. Follow these steps to set it up:

1. Clone this repository or download the files
2. Set up a virtual environment:
   ```
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `.\venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running the Application

1. Activate the virtual environment as described above
2. Run the Streamlit application:
   ```
   streamlit run fcm_ui.py
   ```
3. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)
4. Enter customer details in the form on the left side of the screen
5. Click "Analyze My Customer Profile" to see which customer segments they belong to and their membership degrees

### Retraining the Model

If you want to retrain the FCM model with 4 clusters:

1. Activate the virtual environment
2. Run the training script:
   ```
   python train_fcm_model.py
   ```
3. This will create a new `fcm_model.pkl` file and generate fresh visualizations

### Checking Model Configuration

To verify the model configuration:

1. Activate the virtual environment
2. Run the check script:
   ```
   python check_model.py
   ```
3. This will display information about the current model's configuration

## Customer Segments

The application uses 4 distinct customer segments:

1. **High-Value Loyal Customers**: Customers who spend more, purchase frequently, and give high ratings
2. **Price-Sensitive Occasional Shoppers**: Customers with average spending but longer periods between purchases
3. **New Potential High-Value Customers**: Recent customers with high initial spend but limited history
4. **At-Risk Customers**: Previously active customers who haven't purchased recently

Each customer receives a membership score for each segment (values between 0 and 1), showing how strongly they belong to different customer groups.

## How Fuzzy C-Means Differs from Traditional Segmentation

Traditional customer segmentation (like K-means) assigns each customer to exactly one segment. Fuzzy C-Means recognizes that customers often share characteristics with multiple segments, providing:

1. **Nuanced Understanding**: See how strongly a customer belongs to each segment
2. **Better Marketing Decisions**: Target customers with higher membership in specific segments
3. **Transition Tracking**: Monitor how customers move between segments over time
4. **Personalization**: Customize experiences based on membership degrees

## Technical Details

- The application uses Fuzzy C-Means clustering from the scikit-fuzzy library
- Data is standardized before clustering to ensure all features have equal weight
- Membership degrees are calculated for each customer across all segments
- The model uses the following features:
  - Age
  - Total Spend
  - Items Purchased
  - Average Rating
  - Days Since Last Purchase

## Development Workflow

1. **Data Analysis** (`Pre_Understanding.ipynb`)
   - Initial exploration of the dataset
   - Feature analysis and selection
   - Understanding data distributions

2. **Model Development** (`train_fcm_model.py`)
   - Training the FCM model
   - Generating visualizations
   - Saving the trained model

3. **Model Verification** (`check_model.py`)
   - Ensuring model configuration
   - Validating cluster structure
   - Checking data distributions

4. **Application Development** (`fcm_ui.py`)
   - Building the interactive interface
   - Implementing prediction logic
   - Creating visualizations
