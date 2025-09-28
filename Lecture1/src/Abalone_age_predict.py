# Import required libraries:
# - numpy: For numerical operations and array manipulation
# - torch: Core library for building and training neural networks
# - matplotlib.pyplot: For plotting training curves and evaluation results
# - torch.nn.functional: For neural network functions (e.g., loss calculation)
# - sklearn modules: For data splitting, preprocessing, and evaluation metrics
# - os: For file path handling
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os

# ------------------------------------------------------------------------------
# Data Loading and Preprocessing Function
# ------------------------------------------------------------------------------
def load_and_preprocess_data(file_path=None, use_onehot=True, handle_outliers=True):
    """
    Load and preprocess the Abalone age prediction dataset.
    
    Parameters:
        file_path (str, optional): Path to the dataset file. If None, uses default parent directory path.
        use_onehot (bool, optional): Whether to use One-Hot Encoding for sex feature (True) or numeric encoding (False).
        handle_outliers (bool, optional): Whether to remove outliers using the 3σ rule (True/False).
    
    Returns:
        X_train (np.ndarray): Preprocessed training features
        X_test (np.ndarray): Preprocessed testing features
        Y_train (np.ndarray): Training age labels
        Y_test (np.ndarray): Testing age labels
        train_data (np.ndarray): Combined training features + labels (for batch processing)
        test_data (np.ndarray): Combined testing features + labels (for batch processing)
        input_dim (int): Number of input features (depends on encoding: 11 for one-hot, 9 for numeric)
    """
    # Initialize empty lists to store features (X) and labels (Y)
    data_X = []
    data_Y = []
    
    # Set default file path if not provided (looks for "data/AbaloneAgePrediction.txt" in parent directory)
    if file_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))  # Get current script directory
        parent_dir = os.path.dirname(current_dir)                # Get parent directory of current script
        data_dir = os.path.join(parent_dir, "data")              # Create path to "data" folder
        file_path = os.path.join(data_dir, "AbaloneAgePrediction.txt")  # Full path to dataset
    
    # Read raw data line by line (dataset format: sex, 8 numeric features, age)
    with open(file_path, 'r') as f:
        for line in f:
            line_components = line.strip().split(',')  # Split line by commas
            data_X.append(line_components[:-1])       # First 9 elements: sex + 8 numeric features
            data_Y.append(line_components[-1])        # Last element: age label
    
    # Convert to numpy arrays and separate sex (categorical) and numeric features
    sex_features = np.array([[x[0]] for x in data_X])  # Extract sex feature (M/F/I) as 2D array
    numeric_features = np.array([x[1:] for x in data_X], dtype=np.float32)  # 8 numeric features (float32)
    data_Y = np.array(data_Y, dtype=np.float32)  # Convert age labels to float32
    
    # Encode sex feature (categorical → numerical)
    if use_onehot:
        # One-Hot Encoder: Converts sex (M/F/I) to 3 binary columns (e.g., M → [1,0,0])
        encoder = OneHotEncoder(sparse_output=False, categories=[['M', 'F', 'I']])  # Explicit category order
        encoded_sex = encoder.fit_transform(sex_features)  # Fit encoder and transform sex feature
        data_X = np.hstack((encoded_sex, numeric_features))  # Combine: 3 (one-hot) + 8 (numeric) = 11 features
    else:
        # Numeric encoding: Map sex to integers (M→0, F→1, I→2)
        sex_map = {'M': 0, 'F': 1, 'I': 2}
        encoded_sex = np.array([[sex_map[x[0]]] for x in data_X], dtype=np.float32)  # 1 numeric sex column
        data_X = np.hstack((encoded_sex, numeric_features))  # Combine: 1 (numeric) + 8 (numeric) = 9 features
    
    # Remove outliers using 3σ rule (only applied to numeric features)
    if handle_outliers:
        # Iterate over each numeric feature column
        for col in range(numeric_features.shape[1]):
            mean = np.mean(numeric_features[:, col])  # Mean of the current feature
            std = np.std(numeric_features[:, col])    # Standard deviation of the current feature
            # Create mask: keep samples where feature value is within [mean-3σ, mean+3σ]
            mask = np.logical_and(
                numeric_features[:, col] >= mean - 3*std,
                numeric_features[:, col] <= mean + 3*std
            )
            # Filter outliers from numeric features, encoded sex, and labels
            numeric_features = numeric_features[mask]
            encoded_sex = encoded_sex[mask]
            data_Y = data_Y[mask]
        
        # Re-combine encoded sex and filtered numeric features after outlier removal
        data_X = np.hstack((encoded_sex, numeric_features))
    
    # Min-Max Normalization: Scale all features to [0, 1] (avoids feature dominance by large values)
    for feature_idx in range(data_X.shape[1]):
        feature_min = np.min(data_X[:, feature_idx])  # Minimum value of the current feature
        feature_max = np.max(data_X[:, feature_idx])  # Maximum value of the current feature
        if feature_max > feature_min:  # Avoid division by zero (if all values are the same)
            data_X[:, feature_idx] = (data_X[:, feature_idx] - feature_min) / (feature_max - feature_min)
    
    # Split data into training (80%) and testing (20%) sets
    X_train, X_test, Y_train, Y_test = train_test_split(
        data_X, data_Y,
        test_size=0.2,    # 20% of data for testing
        random_state=42,  # Fixed random seed for reproducibility
        shuffle=True      # Shuffle data before splitting
    )
    
    # Combine features and labels into single arrays (easier for batch processing later)
    train_data = np.concatenate((X_train, Y_train.reshape(-1, 1)), axis=1)  # Axis=1: concatenate columns
    test_data = np.concatenate((X_test, Y_test.reshape(-1, 1)), axis=1)
    
    # Return preprocessed data and input feature dimension
    return X_train, X_test, Y_train, Y_test, train_data, test_data, data_X.shape[1]

# ------------------------------------------------------------------------------
# Neural Network Regression Model Definition
# ------------------------------------------------------------------------------
class RegressionModel(torch.nn.Module):
    """
    Fully connected neural network for regression (Abalone age prediction).
    Architecture: Input → Linear(32) → ReLU → Dropout → Linear(16) → ReLU → Linear(1)
    """
    def __init__(self, input_dim):
        """
        Initialize the regression model.
        
        Parameters:
            input_dim (int): Number of input features (from preprocessing step)
        """
        super(RegressionModel, self).__init__()  # Call parent class (torch.nn.Module) constructor
        # Define fully connected (linear) layers
        self.fc1 = torch.nn.Linear(input_dim, 32)  # Input layer → Hidden layer 1 (32 units)
        self.fc2 = torch.nn.Linear(32, 16)         # Hidden layer 1 → Hidden layer 2 (16 units)
        self.fc3 = torch.nn.Linear(16, 1)          # Hidden layer 2 → Output layer (1 unit: predicted age)
        self.relu = torch.nn.ReLU()                # ReLU activation function (introduces non-linearity)
        self.dropout = torch.nn.Dropout(0.2)       # Dropout layer (20% probability) to prevent overfitting

    def forward(self, x):
        """
        Define the forward pass of the model (data flow through layers).
        
        Parameters:
            x (torch.Tensor): Input tensor (batch of features)
        
        Returns:
            torch.Tensor: Output tensor (batch of predicted ages)
        """
        x = self.fc1(x)       # Pass input through first linear layer
        x = self.relu(x)      # Apply ReLU activation
        x = self.dropout(x)   # Apply dropout (only active during training)
        x = self.fc2(x)       # Pass through second linear layer
        x = self.relu(x)      # Apply ReLU activation
        x = self.fc3(x)       # Pass through output linear layer (no activation for regression)
        return x

# ------------------------------------------------------------------------------
# Training Visualization Utility Class
# ------------------------------------------------------------------------------
class TrainingVisualizer:
    """
    Utility class to record and plot training metrics and evaluation results.
    Tracks training/testing loss and visualizes predictions vs. ground truth.
    """
    def __init__(self):
        """Initialize lists to store metrics for visualization."""
        self.train_sample_counts = []  # Total number of training samples processed
        self.train_mse_losses = []     # MSE loss for each training batch
        self.test_epochs = []          # Epochs where test loss was recorded
        self.test_mse_losses = []      # MSE loss for test set at each epoch

    def record_training_metric(self, sample_count, mse_loss):
        """
        Record training metrics (processed samples and batch MSE loss).
        
        Parameters:
            sample_count (int): Total number of training samples processed so far
            mse_loss (float): MSE loss of the current training batch
        """
        self.train_sample_counts.append(sample_count)
        self.train_mse_losses.append(mse_loss)

    def record_test_metric(self, epoch, mse_loss):
        """
        Record test metrics (epoch and test set MSE loss).
        
        Parameters:
            epoch (int): Current training epoch
            mse_loss (float): MSE loss of the test set at this epoch
        """
        self.test_epochs.append(epoch)
        self.test_mse_losses.append(mse_loss)

    def plot_training_loss(self):
        """Plot MSE loss over the number of processed training samples."""
        plt.figure(figsize=(10, 6))
        plt.title("Training MSE Loss Over Samples")
        plt.xlabel("Number of Processed Training Samples")
        plt.ylabel("MSE Loss")
        plt.plot(self.train_sample_counts, self.train_mse_losses, color='crimson', linewidth=1.5)
        plt.grid(True, alpha=0.3)  # Add light grid for readability
        plt.show()

    def plot_train_test_loss(self):
        """Plot and compare training and test MSE loss over epochs."""
        plt.figure(figsize=(10, 6))
        plt.title("Training vs. Testing MSE Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        
        # Calculate average training loss per epoch (since we have batch-level losses)
        epoch_train_losses = []
        samples_per_epoch = len(self.train_sample_counts) // len(self.test_epochs)  # Batches per epoch
        for i in range(len(self.test_epochs)):
            start_idx = i * samples_per_epoch  # Start index of current epoch's batches
            end_idx = (i + 1) * samples_per_epoch  # End index of current epoch's batches
            epoch_loss = np.mean(self.train_mse_losses[start_idx:end_idx])  # Average loss for the epoch
            epoch_train_losses.append(epoch_loss)
        
        # Plot training and test loss curves
        plt.plot(range(1, len(epoch_train_losses)+1), epoch_train_losses, 
                 color='crimson', linewidth=1.5, label='Training Loss')
        plt.plot(self.test_epochs, self.test_mse_losses, 
                 color='blue', linewidth=1.5, label='Testing Loss')
        plt.legend()  # Show legend
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_inference_results(self, ground_truths, predictions):
        """
        Plot predicted ages vs. ground truth ages (with ideal prediction line y=x).
        
        Parameters:
            ground_truths (np.ndarray): True age values from test set
            predictions (np.ndarray): Predicted age values from the model
        """
        plt.figure(figsize=(10, 6))
        plt.title("Abalone Age: Predictions vs. Ground Truth")
        plt.xlabel("Ground Truth Age")
        plt.ylabel("Predicted Age")
        
        # Define range for ideal prediction line (y=x)
        min_age = min(min(ground_truths), min(predictions))
        max_age = max(max(ground_truths), max(predictions))
        plt.plot([min_age, max_age], [min_age, max_age], color='navy', linestyle='--', label='Ideal Prediction (y=x)')
        
        # Plot scatter of predictions vs. ground truth
        plt.scatter(ground_truths, predictions, color='forestgreen', alpha=0.7, label='Predictions')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_residuals(self, ground_truths, predictions):
        """
        Plot residual analysis: (1) Residuals vs. predictions; (2) Residual distribution histogram.
        
        Residual = Ground Truth - Predicted Value (ideal residuals are centered at 0).
        
        Parameters:
            ground_truths (np.ndarray): True age values from test set
            predictions (np.ndarray): Predicted age values from the model
        """
        residuals = np.array(ground_truths) - np.array(predictions)  # Calculate residuals
        plt.figure(figsize=(12, 5))
        
        # Subplot 1: Residuals vs. predicted values
        plt.subplot(1, 2, 1)
        plt.scatter(predictions, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')  # Horizontal line at y=0 (ideal residual value)
        plt.xlabel("Predicted Age")
        plt.ylabel("Residual (True - Predicted)")
        plt.title("Residuals vs Predictions")
        
        # Subplot 2: Histogram of residual distribution
        plt.subplot(1, 2, 2)
        plt.hist(residuals, bins=15, edgecolor='black')  # 15 bins for smooth distribution
        plt.xlabel("Residual Value")
        plt.ylabel("Frequency")
        plt.title("Distribution of Residuals")
        
        plt.tight_layout()  # Adjust spacing between subplots
        plt.show()

# ------------------------------------------------------------------------------
# Model Training Function (with Early Stopping and Learning Rate Scheduling)
# ------------------------------------------------------------------------------
def train_model(model, device, train_data, test_data, input_dim):
    """
    Train the regression model with early stopping (prevents overfitting) and learning rate scheduling.
    
    Parameters:
        model (RegressionModel): Initialized regression model
        device (torch.device): Device to train on (CPU or GPU)
        train_data (np.ndarray): Combined training features + labels
        test_data (np.ndarray): Combined testing features + labels
        input_dim (int): Number of input features (matches model's input dimension)
    
    Returns:
        model (RegressionModel): Trained model (loaded with best weights from early stopping)
        visualizer (TrainingVisualizer): Object with recorded training/test metrics for visualization
    """
    print("Starting model training...")
    model.train()  # Set model to training mode (enables dropout)
    
    # Training hyperparameters
    EPOCHS = 200          # Maximum number of training epochs
    BATCH_SIZE = 64       # Number of samples per training batch
    LEARNING_RATE = 0.001 # Initial learning rate
    patience = 15         # Early stopping: Stop if test loss doesn't improve for N epochs
    best_test_loss = float('inf')  # Track the best (lowest) test loss
    patience_counter = 0   # Counter for early stopping (resets when test loss improves)
    last_lr = LEARNING_RATE  # Track previous learning rate to detect updates
    
    # Prepare test data as PyTorch tensors (moved to target device)
    test_features = torch.tensor(test_data[:, :input_dim], dtype=torch.float32, device=device)
    test_labels = torch.tensor(test_data[:, -1:], dtype=torch.float32, device=device)
    
    # Initialize optimizer and learning rate scheduler
    # - Optimizer: SGD with momentum (0.9) for stable convergence
    # - Scheduler: ReduceLROnPlateau → Decreases LR when test loss plateaus
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5  # Halve LR after 5 epochs of no improvement
    )
    
    # Initialize visualizer to track training metrics
    visualizer = TrainingVisualizer()
    total_processed_samples = 0  # Track total samples processed across all batches/epochs
    
    # Main training loop (iterate over epochs)
    for epoch in range(EPOCHS):
        # Shuffle training data at the start of each epoch (prevents order bias)
        np.random.shuffle(train_data)
        # Split shuffled training data into mini-batches
        mini_batches = [
            train_data[batch_start:batch_start+BATCH_SIZE] 
            for batch_start in range(0, len(train_data), BATCH_SIZE)
        ]
        
        # Iterate over each mini-batch
        for batch_idx, batch in enumerate(mini_batches):
            # Extract features and labels from the current batch
            batch_features = batch[:, :input_dim].astype(np.float32)  # Features (first N columns)
            batch_labels = batch[:, -1:].astype(np.float32)           # Labels (last column, reshape to 2D)
            
            # Convert numpy arrays to PyTorch tensors (moved to target device)
            features_tensor = torch.tensor(batch_features, device=device)
            labels_tensor = torch.tensor(batch_labels, device=device)
            
            # 1. Forward pass: Compute predicted ages and MSE loss
            predictions = model(features_tensor)
            mse_loss = F.mse_loss(predictions, labels_tensor)  # MSE = mean((y_pred - y_true)²)
            
            # 2. Backward pass: Compute gradients and update model weights
            optimizer.zero_grad()  # Reset gradients (prevents accumulation)
            mse_loss.backward()    # Compute gradients via backpropagation
            optimizer.step()       # Update weights using gradients
            
            # 3. Record training metrics (samples processed and batch loss)
            batch_size_actual = batch_features.shape[0]  # Actual batch size (last batch may be smaller)
            total_processed_samples += batch_size_actual
            visualizer.record_training_metric(total_processed_samples, mse_loss.item())
            
            # Print training progress (every 30 batches, every 20 epochs)
            if batch_idx % 30 == 0 and epoch % 20 == 0:
                print(f"Epoch: {epoch:3d} | Batch: {batch_idx:3d} | MSE Loss: {mse_loss.item():.5f}")
        
        # 4. Evaluate model on test set after each epoch (no gradient computation)
        model.eval()  # Set model to evaluation mode (disables dropout)
        with torch.no_grad():  # Disable gradient calculation to save memory
            test_preds = model(test_features)
            test_loss = F.mse_loss(test_preds, test_labels).item()  # Compute test MSE loss
            visualizer.record_test_metric(epoch + 1, test_loss)     # Record test metrics (epoch starts at 0)
        
        # 5. Adjust learning rate based on test loss (scheduler step)
        scheduler.step(test_loss)
        current_lr = optimizer.param_groups[0]['lr']  # Get current learning rate
        if current_lr != last_lr:
            print(f"Learning rate adjusted from {last_lr:.6f} to {current_lr:.6f}")
            last_lr = current_lr
        
        # 6. Early stopping check: Stop if test loss doesn't improve
        if test_loss < best_test_loss:
            best_test_loss = test_loss  # Update best test loss
            torch.save(model.state_dict(), "best_abalone_model.pth")  # Save best model weights
            patience_counter = 0  # Reset counter (loss improved)
        else:
            patience_counter += 1  # Increment counter (loss didn't improve)
            # Stop training if counter reaches patience threshold
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        model.train()  # Reset model to training mode for next epoch
    
    # Load the best model weights (from early stopping) before returning
    model.load_state_dict(torch.load("best_abalone_model.pth"))
    return model, visualizer

# ------------------------------------------------------------------------------
# Model Evaluation Function
# ------------------------------------------------------------------------------
def run_evaluation(model, device, test_data, input_dim, batch_size=30):
    """
    Evaluate the trained model on the test set and print key metrics (MSE, RMSE, MAE, R²).
    
    Parameters:
        model (RegressionModel): Trained regression model
        device (torch.device): Device to run evaluation on (CPU or GPU)
        test_data (np.ndarray): Combined testing features + labels
        input_dim (int): Number of input features
        batch_size (int, optional): Number of test samples to display detailed predictions for
    
    Returns:
        ground_truths (np.ndarray): Subset of true age values (for visualization)
        predictions (np.ndarray): Corresponding predicted age values (for visualization)
    """
    print("\nStarting evaluation on test data...")
    model.eval()  # Set model to evaluation mode
    
    # Extract test features and labels from combined test data
    test_features = test_data[:, :input_dim].astype(np.float32)
    test_labels = test_data[:, -1].astype(np.float32)  # 1D array (true ages)
    
    # Use a subset of test samples for detailed prediction display
    test_features_subset = test_features[:batch_size]
    test_labels_subset = test_labels[:batch_size]
    
    # Run model inference on the subset (no gradient computation)
    with torch.no_grad():
        features_tensor = torch.tensor(test_features_subset, device=device)
        predictions = model(features_tensor).cpu().numpy().flatten()  # Convert to 1D numpy array
    
    # Calculate evaluation metrics (using subset for detailed analysis)
    mse = mean_squared_error(test_labels_subset, predictions)  # Mean Squared Error
    rmse = np.sqrt(mse)                                       # Root Mean Squared Error (interpretable in age units)
    mae = mean_absolute_error(test_labels_subset, predictions)# Mean Absolute Error
    r2 = r2_score(test_labels_subset, predictions)            # R² Score (0 = no explanation, 1 = perfect explanation)
    
    # Print detailed predictions for the subset
    print("\nSample Predictions vs. Ground Truth:")
    for idx in range(min(batch_size, len(predictions))):
        pred = predictions[idx]
        gt = test_labels_subset[idx]
        print(f"Sample {idx+1:2d} | Predicted: {pred:.2f} | Actual: {gt:.2f} | Error: {abs(pred-gt):.2f}")
    
    # Print summary of evaluation metrics
    print("\nEvaluation Metrics (Test Subset):")
    print(f"MSE: {mse:.5f}")
    print(f"RMSE: {rmse:.5f} (Average Prediction Error in Age Units)")
    print(f"MAE: {mae:.5f} (Average Absolute Error in Age Units)")
    print(f"R² Score: {r2:.5f} (Proportion of Variance Explained by Model)")
    
    # Return subset of ground truths and predictions for visualization
    return test_labels_subset, predictions

# ------------------------------------------------------------------------------
# Main Execution (Run When Script is Called Directly)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Step 1: Set device (use GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for training/evaluation: {device}")
    
    # Step 2: Load and preprocess data
    # - use_onehot=True: Encode sex as 3 one-hot columns
    # - handle_outliers=True: Remove outliers using 3σ rule
    X_train, X_test, Y_train, Y_test, train_data, test_data, input_dim = load_and_preprocess_data(
        use_onehot=True,
        handle_outliers=True
    )
    print(f"Data preprocessing complete. Input feature dimension: {input_dim}")
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    
    # Step 3: Initialize the regression model and move it to the target device
    model = RegressionModel(input_dim).to(device)
    
    # Step 4: Train the model (with early stopping and LR scheduling)
    model, visualizer = train_model(model, device, train_data, test_data, input_dim)
    
    # Step 5: Visualize training progress (loss curves)
    visualizer.plot_training_loss()
    visualizer.plot_train_test_loss()
    
    # Step 6: Evaluate the trained model on test data
    ground_truths, predictions = run_evaluation(model, device, test_data, input_dim, batch_size=30)
    
    # Step 7: Visualize evaluation results (predictions vs. truth, residuals)
    visualizer.plot_inference_results(ground_truths, predictions)
    visualizer.plot_residuals(ground_truths, predictions)