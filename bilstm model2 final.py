import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.regularizers import l2
from kerastuner import HyperModel, RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/kaggle/input/subset5/subset_data5.csv'
data = pd.read_csv(file_path)

# Convert 'time' column to datetime
data['time'] = pd.to_datetime(data['time'])

# Extract useful time-based features
data['year'] = data['time'].dt.year
data['month'] = data['time'].dt.month
data['day'] = data['time'].dt.day
data['hour'] = data['time'].dt.hour
data['minute'] = data['time'].dt.minute
data['second'] = data['time'].dt.second

# Encode the 'label' column
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])

# Define features and target variable
features = ['lat', 'lon', 'alt', 'time_diff', 'distance', 'speed', 'acceleration', 'bearing', 'pitch', 'year', 'month', 'day', 'hour', 'minute', 'second']
X = data[features]
y = data['label']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Reshape data for BiLSTM input
X_train_bilstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val_bilstm = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

# Define the BiLSTM model with L2 regularization
class BiLSTMHyperModel(HyperModel):

    """
    A class to build and compile a BiLSTM model with hyperparameter tuning.

    Attributes:
        None
    """
    
    def build(self, hp):

        """
        Builds the BiLSTM model with hyperparameter tuning.

        Args:
            hp (kerastuner.engine.hyperparameters.HyperParameters): Hyperparameter tuning object.

        Returns:
            model (tf.keras.Sequential): Compiled BiLSTM model.
        """
        
        # Instantiate a Sequential model
        model = Sequential()

        # Add a Bidirectional LSTM layer with hyperparameter tuning for the number of units and L2 regularization
        model.add(Bidirectional(LSTM(units=hp.Int('units', min_value=32, max_value=256, step=32),
                                     activation='relu',
                                     input_shape=(X_train_bilstm.shape[1], X_train_bilstm.shape[2]),
                                     kernel_regularizer=l2(hp.Float('l2', min_value=1e-4, max_value=1e-2, sampling='LOG')))))

        # Add a Dropout layer with hyperparameter tuning for the dropout rate
        model.add(Dropout(rate=hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)))

        # Add a Dense output layer with softmax activation for classification
        model.add(Dense(units=len(label_encoder.classes_), activation='softmax'))

        # Compile the model with Adam optimizer, sparse categorical crossentropy loss, and accuracy metric
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Float('lr', min_value=1e-4, max_value=1e-2, sampling='LOG')),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        return model
    # Return the compiled model


# Initialize a list to store results
results = []

# Initialize the hypermodel
hypermodel = BiLSTMHyperModel()

# Initialize the tuner
tuner = RandomSearch(
    hypermodel,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=2,
    directory='bilstm_hyperparameter_tuning',
    project_name='trajectory_prediction'
)

# Perform hyperparameter tuning
for trial in range(5):
    tuner.search(X_train_bilstm, y_train, epochs=5, validation_data=(X_val_bilstm, y_val))
    
    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    # Build the best BiLSTM model
    best_model = tuner.hypermodel.build(best_hps)
    
    # Train the best BiLSTM model
    history = best_model.fit(X_train_bilstm, y_train, epochs=10, validation_data=(X_val_bilstm, y_val))
    
    # Record results
    
    # Record results
    # Append a dictionary of results for the current trial to the results list
    results.append({
        'Trial': trial + 1,  # Record the trial number (1-based index)
        'Best Hyperparameters': best_hps.values,  # Record the best hyperparameters found in the current trial
        'Training Accuracy': history.history['accuracy'][-1],  # Record the training accuracy at the last epoch
        'Validation Accuracy': history.history['val_accuracy'][-1],  # Record the validation accuracy at the last epoch
        'Training Loss': history.history['loss'][-1],  # Record the training loss at the last epoch
        'Validation Loss': history.history['val_loss'][-1]  # Record the validation loss at the last epoch
    })
    
    # Print or save the results after each trial
    print(f"Trial {trial + 1}:")  # Print the current trial number
    print(f"Best Hyperparameters: {best_hps.values}")  # Print the best hyperparameters found in the current trial
    print(f"Training Accuracy: {history.history['accuracy'][-1]}")  # Print the training accuracy at the last epoch
    print(f"Validation Accuracy: {history.history['val_accuracy'][-1]}")  # Print the validation accuracy at the last epoch
    print(f"Training Loss: {history.history['loss'][-1]}")  # Print the training loss at the last epoch
    print(f"Validation Loss: {history.history['val_loss'][-1]}")  # Print the validation loss at the last epoch
    print("=" * 50)  # Print a separator line for readability


# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save results to a CSV file
results_df.to_csv('bilstm_hyperparameter_tuning_results.csv', index=False)

# Plot training and validation loss and accuracy for the best model
best_trial_idx = results_df['Validation Accuracy'].idxmax()
best_trial_history = results[best_trial_idx]

# Recreate the best hyperparameters
best_hps = HyperParameters()  # Initialize a new HyperParameters object

# Loop through each key-value pair in the best hyperparameters dictionary
for key, value in best_trial_history['Best Hyperparameters'].items():
    # Check if the key is 'units', 'l2', 'dropout', or 'lr'
    if 'units' in key or 'l2' in key or 'dropout' in key or 'lr' in key:
        # If the key is 'units', set an integer hyperparameter with the appropriate range and default value
        if 'units' in key:
            best_hps.Int(key, min_value=32, max_value=256, step=32, default=value)
        # If the key is 'l2', set a float hyperparameter with logarithmic sampling and the appropriate range and default value
        elif 'l2' in key:
            best_hps.Float(key, min_value=1e-4, max_value=1e-2, sampling='LOG', default=value)
        # If the key is 'dropout', set a float hyperparameter with linear sampling and the appropriate range and default value
        elif 'dropout' in key:
            best_hps.Float(key, min_value=0.2, max_value=0.5, step=0.1, default=value)
        # If the key is 'lr' (learning rate), set a float hyperparameter with logarithmic sampling and the appropriate range and default value
        elif 'lr' in key:
            best_hps.Float(key, min_value=1e-4, max_value=1e-2, sampling='LOG', default=value)


# Build and train the model with the best hyperparameters
best_model = hypermodel.build(best_hps)
best_history = best_model.fit(X_train_bilstm, y_train, epochs=10, validation_data=(X_val_bilstm, y_val))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(best_history.history['loss'], label='Training Loss')
plt.plot(best_history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(best_history.history['accuracy'], label='Training Accuracy')
plt.plot(best_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Print training and validation accuracy for the best model
train_acc = best_history.history['accuracy'][-1]
val_acc = best_history.history['val_accuracy'][-1]
print(f'Best Model Training Accuracy: {train_acc}')
print(f'Best Model Validation Accuracy: {val_acc}')
