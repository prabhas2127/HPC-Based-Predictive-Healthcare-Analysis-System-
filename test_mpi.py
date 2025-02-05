from mpi4py import MPI
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# MPI Initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Start measuring total execution time
start_total_time = time.time()

# Load the dataset (only in rank 0)
data_path = r'C:\Users\user\Desktop\ML_module\project\kagg\Training.csv'
if rank == 0:
    training = pd.read_csv(data_path)
else:
    training = None

# Broadcast dataset to all processes
training = comm.bcast(training, root=0)

# Function to augment the data
def augment_data(df, num_copies):
    augmented_data = df.copy()
    
    for _ in range(num_copies):
        for column in df.columns[:-1]:  # Exclude 'prognosis'
            if df[column].dtype == 'int64':
                flip_mask = np.random.rand(len(augmented_data)) < 0.1
                augmented_data.loc[flip_mask, column] = 1 - augmented_data.loc[flip_mask, column]
    
    return augmented_data

# Determine data augmentation per process
original_size = len(training)
target_size = 10000
num_copies = target_size // (original_size * size)  # Split across MPI processes

# Each process augments its portion
df_split = np.array_split(training, size)[rank]
augmented_data = augment_data(df_split, num_copies)

# Gather augmented data at rank 0
augmented_data_all = comm.gather(augmented_data, root=0)

if rank == 0:
    expanded_df = pd.concat(augmented_data_all, ignore_index=True)
    expanded_df.to_csv(r'C:\Users\user\Desktop\ML_module\project\kagg\Expanded_par_Training.csv', index=False)
else:
    expanded_df = None

# Broadcast expanded dataset
temp_data = comm.bcast(expanded_df, root=0)

# Label encoding - Fit LabelEncoder in rank 0 and broadcast it
if rank == 0:
    label_encoder = LabelEncoder()
    label_encoder.fit(temp_data['prognosis'])  # Fit only once on the entire 'prognosis' column
else:
    label_encoder = None

# Broadcast the fitted LabelEncoder to all processes
label_encoder = comm.bcast(label_encoder, root=0)

# Encode labels
y_encoded = label_encoder.transform(temp_data['prognosis'])  # Use transform after fitting
X = temp_data.drop(['prognosis'], axis=1)

# Split dataset (same for all processes)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train Random Forest Classifier
start_train_time = time.time()
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
train_time = time.time() - start_train_time

# Evaluate model
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Gather accuracy metrics at rank 0
accuracy_all = comm.gather(accuracy, root=0)
train_time_all = comm.gather(train_time, root=0)

# Prepare prediction data (e.g., df2 should be a test sample from X_test)
df2 = pd.DataFrame(columns=X_test.columns)
df2 = pd.concat([df2, X_test.iloc[[7]]], axis=0, ignore_index=True)  # Use test set's eighth row for prediction
df2 = df2.fillna(0)  # Ensure no missing values in df2

# Make prediction using the trained model
m = rf.predict(df2)

# Inverse transform the predicted labels using the fitted label_encoder
dis = label_encoder.inverse_transform(m)

# Gathering results at rank 0
if rank == 0:
    # Calculate average accuracy and training time across all processes
    avg_accuracy = np.mean(accuracy_all)
    avg_train_time = np.mean(train_time_all)
    
    # Print model evaluation results
    print("\n--- Parallel Model Evaluation Results ---\n")
    print(f"Average Accuracy: {avg_accuracy:.4f}\n")
    print(f"Average Training Time: {avg_train_time:.4f} seconds\n")
    print("\nClassification Report:\n\n", class_report)

    # Print prediction for rank 0
    print('You should do test of : ', dis)
    print('\n\n')
else:
    # Print result from other ranks (optional)
    print(f"Rank {rank} - Accuracy: {accuracy:.4f}, Training Time: {train_time:.4f} seconds")
    print('\n\n')
