import pickle
import os

print("Checking FCM model configuration...")
model_path = 'fcm_model.pkl'

if not os.path.exists(model_path):
    print(f"Model file {model_path} not found.")
else:
    print(f"Model file {model_path} exists. Size: {os.path.getsize(model_path) / 1024:.2f} KB")
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
            print(f"Model loaded successfully.")
            
            # Check key properties
            n_clusters = model.get('n_clusters')
            print(f"Number of clusters: {n_clusters}")
            
            features = model.get('features')
            print(f"Features used: {features}")
            
            if 'centers' in model:
                centers = model['centers']
                print(f"Centers shape: {centers.shape}")
                print(f"This confirms the model has {centers.shape[0]} clusters.")
            
            # Check if membership matrix exists
            if 'u' in model:
                u_matrix = model['u']
                print(f"Membership matrix shape: {u_matrix.shape}")
                print(f"This confirms memberships for {u_matrix.shape[0]} clusters across {u_matrix.shape[1]} samples.")
            
            # Check if cluster labels exist and their distribution
            if 'cluster_labels' in model:
                labels = model['cluster_labels']
                unique_labels = set(labels)
                print(f"Unique cluster labels: {sorted(unique_labels)}")
                
                # Count samples in each cluster
                from collections import Counter
                label_counts = Counter(labels)
                for label, count in sorted(label_counts.items()):
                    print(f"  Cluster {label}: {count} samples ({count/len(labels)*100:.1f}%)")
            
    except Exception as e:
        print(f"Error loading model: {str(e)}")

# Check visualization files
membership_viz = 'membership_distribution.png'
characteristics_viz = 'cluster_characteristics.png'

print("\nChecking visualization files:")
if os.path.exists(membership_viz):
    print(f"Membership distribution visualization exists. Size: {os.path.getsize(membership_viz) / 1024:.2f} KB")
else:
    print(f"Membership distribution visualization not found.")

if os.path.exists(characteristics_viz):
    print(f"Cluster characteristics visualization exists. Size: {os.path.getsize(characteristics_viz) / 1024:.2f} KB")
else:
    print(f"Cluster characteristics visualization not found.")

print("\nCheck complete!") 