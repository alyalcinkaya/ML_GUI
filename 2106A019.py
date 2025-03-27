import sys
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QTabWidget, QFormLayout, QPushButton, QLabel, 
                           QComboBox, QFileDialog, QSpinBox, QDoubleSpinBox,
                           QGroupBox, QScrollArea, QTextEdit, QStatusBar,
                           QProgressBar, QCheckBox, QGridLayout, QMessageBox,
                           QDialog, QLineEdit)
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn import datasets, preprocessing, model_selection
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC , SVR 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (mean_absolute_error, accuracy_score, 
                             mean_squared_error, confusion_matrix)
import tensorflow as tf 
from keras import layers, models, optimizers

class MLCourseGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Machine Learning Course GUI")
        self.setGeometry(100, 100, 1400, 800)
        
        # Initialize main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)
        
        # Initialize data containers
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.current_model = None
        
        # Neural network configuration
        self.layer_config = []
        
        # Create components
        self.create_data_section()
        self.create_tabs()
        self.create_visualization()
        self.create_status_bar()

    def load_dataset(self):
        """Load selected dataset"""
        try:
            dataset_name = self.dataset_combo.currentText()
            
            if dataset_name == "Load Custom Dataset":
                return
            
            # Load selected dataset
            if dataset_name == "Iris Dataset":
                data = datasets.load_iris()
            elif dataset_name == "Breast Cancer Dataset":
                data = datasets.load_breast_cancer()
            elif dataset_name == "Digits Dataset":
                data = datasets.load_digits()
            elif dataset_name == "Boston Housing Dataset":
                data = datasets.load_boston()
            elif dataset_name == "MNIST Dataset":
                (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
                self.X_train, self.X_test = X_train, X_test
                self.y_train, self.y_test = y_train, y_test
                self.status_bar.showMessage(f"Loaded {dataset_name}")
                return
            
            # Split data
            test_size = self.split_spin.value()
            self.X_train, self.X_test, self.y_train, self.y_test = \
                model_selection.train_test_split(data.data, data.target, 
                                              test_size=test_size, 
                                              random_state=42)
            
            # Apply scaling if selected
            self.apply_scaling()
            
            self.status_bar.showMessage(f"Loaded {dataset_name}")
            
        except Exception as e:
            self.show_error(f"Error loading dataset: {str(e)}")
    
    def load_custom_data(self):
        """Load custom dataset from CSV file"""
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Load Dataset",
                "",
                "CSV files (*.csv)"
            )
            
            if file_name:
                # Load data
                data = pd.read_csv(file_name)
                
                # Ask user to select target column
                target_col = self.select_target_column(data.columns)
                
                if target_col:
                    X = data.drop(target_col, axis=1)
                    y = data[target_col]
                    
                    # Split data
                    test_size = self.split_spin.value()
                    self.X_train, self.X_test, self.y_train, self.y_test = \
                        model_selection.train_test_split(X, y, 
                                                      test_size=test_size, 
                                                      random_state=42)
                    
                    # Apply scaling if selected
                    self.apply_scaling()
                    
                    self.status_bar.showMessage(f"Loaded custom dataset: {file_name}")
                    
        except Exception as e:
            self.show_error(f"Error loading custom dataset: {str(e)}")
    
    def select_target_column(self, columns):
        """Dialog to select target column from dataset"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Target Column")
        layout = QVBoxLayout(dialog)
        
        combo = QComboBox()
        combo.addItems(columns)
        layout.addWidget(combo)
        
        btn = QPushButton("Select")
        btn.clicked.connect(dialog.accept)
        layout.addWidget(btn)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return combo.currentText()
        return None
    
    def apply_scaling(self):
        """Apply selected scaling method to the data"""
        scaling_method = self.scaling_combo.currentText()
        
        if scaling_method != "No Scaling":
            try:
                if scaling_method == "Standard Scaling":
                    scaler = preprocessing.StandardScaler()
                elif scaling_method == "Min-Max Scaling":
                    scaler = preprocessing.MinMaxScaler()
                elif scaling_method == "Robust Scaling":
                    scaler = preprocessing.RobustScaler()
                elif scaling_method == "Normalize":
                    scaler = preprocessing.Normalizer()

                self.X_train = scaler.fit_transform(self.X_train)
                self.X_test = scaler.transform(self.X_test)
                
            except Exception as e:
                self.show_error(f"Error applying scaling: {str(e)}")

    def create_data_section(self):
        """Create the data loading and preprocessing section"""
        data_group = QGroupBox("Data Management")
        data_layout = QHBoxLayout()
        
        # Dataset selection
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems([
            "Load Custom Dataset",
            "Iris Dataset",
            "Breast Cancer Dataset",
            "Digits Dataset",
            "Boston Housing Dataset",
            "MNIST Dataset"
        ])
        self.dataset_combo.currentIndexChanged.connect(self.load_dataset)
        
        # Data loading button
        self.load_btn = QPushButton("Load Data")
        self.load_btn.clicked.connect(self.load_custom_data)
        
        # Preprocessing options
        self.scaling_combo = QComboBox()
        self.scaling_combo.addItems([
            "No Scaling",
            "Standard Scaling",
            "Min-Max Scaling",
            "Robust Scaling",
            "Normalize"
        ])
        
        # Train-test split options
        self.split_spin = QDoubleSpinBox()
        self.split_spin.setRange(0.1, 0.9)
        self.split_spin.setValue(0.2)
        self.split_spin.setSingleStep(0.1)
        
        # Add widgets to layout
        data_layout.addWidget(QLabel("Dataset:"))
        data_layout.addWidget(self.dataset_combo)
        data_layout.addWidget(self.load_btn)
        data_layout.addWidget(QLabel("Scaling:"))
        data_layout.addWidget(self.scaling_combo)
        data_layout.addWidget(QLabel("Test Split:"))
        data_layout.addWidget(self.split_spin)
        
        data_group.setLayout(data_layout)
        self.layout.addWidget(data_group)
    
    def create_tabs(self):
        """Create tabs for different ML topics"""
        self.tab_widget = QTabWidget()
        
        # Create individual tabs
        tabs = [
            ("Classical ML", self.create_classical_ml_tab),
            ("Deep Learning", self.create_deep_learning_tab),
            ("Dimensionality Reduction", self.create_dim_reduction_tab),
            ("Reinforcement Learning", self.create_rl_tab)
        ]
        
        for tab_name, create_func in tabs:
            scroll = QScrollArea()
            tab_widget = create_func()
            scroll.setWidget(tab_widget)
            scroll.setWidgetResizable(True)
            self.tab_widget.addTab(scroll, tab_name)
        
        self.layout.addWidget(self.tab_widget)
    
    def create_classical_ml_tab(self):
        """Create the classical machine learning algorithms tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # Regression section
        regression_group = QGroupBox("Regression")
        regression_layout = QVBoxLayout()
        
        self.regression_loss_combo = QComboBox()
        self.regression_loss_combo.addItems(["MSE", "MAE", "Huber Loss"])
        self.regression_loss_combo.setCurrentText("MSE")  # Default loss
        regression_layout.addWidget(QLabel("Loss Function:"))
        regression_layout.addWidget(self.regression_loss_combo)
        
        # Linear Regression
        lr_group = self.create_algorithm_group(
            "Linear Regression",
            {"fit_intercept": "checkbox",
             "n_jobs": "int",
             "positive": "checkbox",
             "feature_names": "text", 
             "target_name": "text",  
             "test_size": "double"
            }
        )
        regression_layout.addWidget(lr_group)
        
        # Logistic Regression
        logistic_group = self.create_algorithm_group(
            "Logistic Regression",
            {"C": "double",
             "max_iter": "int",
             "multi_class": ["ovr", "multinomial"],
             "penalty": ["l1", "l2", "elasticnet", "none"],
             "solver": ["lbfgs", "newton-cg", "liblinear", "sag", "saga"],
             "class_weight": ["balanced", "none"],
             "random_state": "int",                 
             "verbose": "int",
             "tol": "double",
             "warm_start": "checkbox",
             "feature_names": "text",
             "test_size": "double",
             "l1_ratio": "double"}
        )
        regression_layout.addWidget(logistic_group)

        svr_group = self.create_algorithm_group(
            "Support Vector Regression",
            {
                "C": "double",
                "kernel": ["linear", "rbf", "poly"],
                "epsilon": "double"
            }
        )
        regression_layout.addWidget(svr_group)
        
        regression_group.setLayout(regression_layout)
        layout.addWidget(regression_group, 0, 0)
        
        # Classification section
        classification_group = QGroupBox("Classification")
        classification_layout = QVBoxLayout()
        
        # Add loss function selection
        loss_group = QGroupBox("Loss Function")
        loss_layout = QHBoxLayout()
        self.classification_loss_combo = QComboBox()
        self.classification_loss_combo.addItems(["Cross-Entropy", "Hinge Loss"])
        loss_layout.addWidget(QLabel("Loss Function:"))
        loss_layout.addWidget(self.classification_loss_combo)
        loss_group.setLayout(loss_layout)
        classification_layout.addWidget(loss_group)

        # Naive Bayes
        nb_group = self.create_algorithm_group(
            "Naive Bayes",
            {"var_smoothing": "double",
             "feature_names": "text",
             "priors_type": ["uniform", "custom"],  # Type of prior distribution
             "custom_priors": "text",  # Custom prior probabilities
             "target_name": "text"}
        )
        classification_layout.addWidget(nb_group)
        
        # SVM
        svm_group = self.create_algorithm_group(
            "Support Vector Machine",
            {"C": "double",
             "kernel": ["linear", "rbf", "poly", "sigmoid"],
             "degree": "int",
             "gamma": ["scale", "auto"],
             "coef0": "double",
             "class_weight": ["balanced", "none"],
             "tol": "double",
             "max_iter": "int"}
        )
        classification_layout.addWidget(svm_group)
        
        # Decision Trees
        dt_group = self.create_algorithm_group(
            "Decision Tree",
            {"criterion": ["gini", "entropy"],
             "max_depth": "int",
             "min_samples_split": "int",
             "min_samples_leaf": "int",
             "max_features": ["sqrt", "log2", "none"],
             "random_state": "int",
             "class_weight": ["balanced", "none"]}
        )
        classification_layout.addWidget(dt_group)
        
        # Random Forest
        rf_group = self.create_algorithm_group(
            "Random Forest",
            {"criterion": ["gini", "entropy"],
             "n_estimators": "int",
             "max_depth": "int",
             "min_samples_split": "int",
             "min_samples_leaf": "int",
             "max_features": ["sqrt", "log2", "none"],
             "bootstrap": "checkbox",
             "random_state": "int",
             "class_weight": ["balanced", "none"],
             "max_leaf_nodes": "int",
             "min_impurity_decrease": "double"}
        )
        classification_layout.addWidget(rf_group)
        
        # KNN
        knn_group = self.create_algorithm_group(
            "K-Nearest Neighbors",
            {"n_neighbors": "int",
             "metric": ["euclidean", "manhattan", "minkowski", "chebyshev", "mahalanobis", "custom"],
             "weights": ["uniform", "distance"],
             "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
             "leaf_size": "int"}
        )
        classification_layout.addWidget(knn_group)
        
        classification_group.setLayout(classification_layout)
        layout.addWidget(classification_group, 0, 1)
        
        return widget
    
    def create_dim_reduction_tab(self):
        """Create the dimensionality reduction tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # K-Means section
        kmeans_group = QGroupBox("K-Means Clustering")
        kmeans_layout = QVBoxLayout()
        
        kmeans_params = self.create_algorithm_group(
            "K-Means Parameters",
            {"n_clusters": "int",
             "max_iter": "int",
             "n_init": "int"}
        )
        kmeans_layout.addWidget(kmeans_params)
        
        kmeans_group.setLayout(kmeans_layout)
        layout.addWidget(kmeans_group, 0, 0)
        
        # PCA section
        pca_group = QGroupBox("Principal Component Analysis")
        pca_layout = QVBoxLayout()
        
        pca_params = self.create_algorithm_group(
            "PCA Parameters",
            {"n_components": "int",
             "whiten": "checkbox"}
        )
        pca_layout.addWidget(pca_params)
        
        pca_group.setLayout(pca_layout)
        layout.addWidget(pca_group, 0, 1)
        
        return widget
    
    def create_rl_tab(self):
        """Create the reinforcement learning tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # Environment selection
        env_group = QGroupBox("Environment")
        env_layout = QVBoxLayout()
        
        self.env_combo = QComboBox()
        self.env_combo.addItems([
            "CartPole-v1",
            "MountainCar-v0",
            "Acrobot-v1"
        ])
        env_layout.addWidget(self.env_combo)
        
        env_group.setLayout(env_layout)
        layout.addWidget(env_group, 0, 0)
        
        # RL Algorithm selection
        algo_group = QGroupBox("RL Algorithm")
        algo_layout = QVBoxLayout()
        
        self.rl_algo_combo = QComboBox()
        self.rl_algo_combo.addItems([
            "Q-Learning",
            "SARSA",
            "DQN"
        ])
        algo_layout.addWidget(self.rl_algo_combo)
        
        algo_group.setLayout(algo_layout)
        layout.addWidget(algo_group, 0, 1)
        
        return widget
    
    
    def create_visualization(self):
        """Create the visualization section"""
        viz_group = QGroupBox("Visualization")
        viz_layout = QHBoxLayout()
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        viz_layout.addWidget(self.canvas)
        
        # Metrics display
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        viz_layout.addWidget(self.metrics_text)
        
        viz_group.setLayout(viz_layout)
        self.layout.addWidget(viz_group)
    
    def create_status_bar(self):
        """Create the status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Add progress bar
        self.progress_bar = QProgressBar()
        self.status_bar.addPermanentWidget(self.progress_bar)

    def display_linear_regression_results(self, model, y_pred):
        """Display linear regression results including coefficients and metrics"""
        try:
            # Clear previous results
            self.metrics_text.clear()
        
            # Calculate metrics
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = model.score(self.X_test, self.y_test)
        
            # Format results text
            results = "Linear Regression Results\n"
            results += "=" * 30 + "\n\n"
        
            # Model coefficients
            results += "Coefficients:\n"
            feature_names = getattr(self, 'feature_names', [f"Feature {i}" for i in range(len(model.coef_))])
            for name, coef in zip(feature_names, model.coef_):
                results += f"{name}: {coef:.4f}\n"
        
            results += f"\nIntercept: {model.intercept_:.4f}\n"
        
            # Model performance metrics
            results += "\nModel Performance:\n"
            results += f"Mean Squared Error (MSE): {mse:.4f}\n"
            results += f"Root Mean Squared Error (RMSE): {rmse:.4f}\n"
            results += f"Mean Absolute Error (MAE): {mae:.4f}\n"
            results += f"R² Score: {r2:.4f}\n"
        
            # Display results
            self.metrics_text.setText(results)
            
            # Update visualization
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.scatter(y_pred, self.y_test, alpha=0.5)
            ax.plot([self.y_test.min(), self.y_test.max()],
                    [self.y_test.min(), self.y_test.max()],
                    'r--', lw=2)
            ax.set_xlabel("Predicted Values")
            ax.set_ylabel("Actual Values")
            ax.set_title("Linear Regression: Predicted vs Actual Values")
            self.canvas.draw()
            
        except Exception as e:
            self.show_error(f"Error displaying linear regression results: {str(e)}")

    def display_logistic_regression_results(self, model, y_pred):
        """Display logistic regression results including metrics"""
        try:
            # Clear previous results
            self.metrics_text.clear()
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            conf_mat = confusion_matrix(self.y_test, y_pred)
            
            # Format results text
            results = "Logistic Regression Results\n"
            results += "=" * 30 + "\n\n"
            
            # Model coefficients (for binary classification)
            if len(model.classes_) == 2:
                results += "Coefficients:\n"
                feature_names = getattr(self, 'feature_names', [f"Feature {i}" for i in range(len(model.coef_[0]))])
                for name, coef in zip(feature_names, model.coef_[0]):
                    results += f"{name}: {coef:.4f}\n"
                results += f"\nIntercept: {model.intercept_[0]:.4f}\n"
            
            # Model performance metrics
            results += "\nModel Performance:\n"
            results += f"Accuracy: {accuracy:.4f}\n\n"
            results += "Confusion Matrix:\n"
            results += str(conf_mat) + "\n"
            
            # Display results
            self.metrics_text.setText(results)
            
            # Update visualization
            self.figure.clear()
            if self.X_test.shape[1] <= 2:  # 2D visualization
                ax = self.figure.add_subplot(111)
                scatter = ax.scatter(self.X_test[:, 0],
                                self.X_test[:, 1] if self.X_test.shape[1] > 1 else np.zeros_like(self.X_test[:, 0]),
                                c=y_pred,
                                cmap='viridis')
                ax.set_xlabel("Feature 1")
                ax.set_ylabel("Feature 2" if self.X_test.shape[1] > 1 else "")
                ax.set_title("Logistic Regression: Classification Results")
                self.figure.colorbar(scatter)
            else:  # PCA visualization for high-dimensional data
                pca = PCA(n_components=2)
                X_test_2d = pca.fit_transform(self.X_test)
                ax = self.figure.add_subplot(111)
                scatter = ax.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_pred, cmap='viridis')
                ax.set_xlabel("First Principal Component")
                ax.set_ylabel("Second Principal Component")
                ax.set_title("Logistic Regression: Classification Results (PCA)")
                self.figure.colorbar(scatter)
            
            self.canvas.draw()
            
        except Exception as e:
            self.show_error(f"Error displaying logistic regression results: {str(e)}")

    def display_svr_results(self, model, y_pred):
        """Display SVR results including metrics and visualization"""
        try:
            # Clear previous results
            self.metrics_text.clear()

            # Calculate metrics
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = model.score(self.X_test, self.y_test)

            # Format results text
            results = "Support Vector Regression Results\n"
            results += "=" * 30 + "\n\n"
            results += f"Kernel: {model.kernel}\n"
            results += f"C: {model.C}\n"
            results += f"Epsilon: {model.epsilon}\n\n"
            results += "Model Performance:\n"
            results += f"Mean Squared Error (MSE): {mse:.4f}\n"
            results += f"Root Mean Squared Error (RMSE): {rmse:.4f}\n"
            results += f"Mean Absolute Error (MAE): {mae:.4f}\n"
            results += f"R² Score: {r2:.4f}\n"

            # Display results
            self.metrics_text.setText(results)

            # Update visualization
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.scatter(self.y_test, y_pred, alpha=0.5)
            ax.plot([self.y_test.min(), self.y_test.max()],
                    [self.y_test.min(), self.y_test.max()],
                    'r--', lw=2)
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            ax.set_title("Support Vector Regression: Predicted vs Actual Values")
            self.canvas.draw()

        except Exception as e:
            self.show_error(f"Error displaying SVR results: {str(e)}")

    def display_naive_bayes_results(self, model, y_pred):
        """Display Naive Bayes results including metrics and visualization"""
        try:
            # Clear previous results
            self.metrics_text.clear()
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            conf_mat = confusion_matrix(self.y_test, y_pred)
            
            # Format results text
            results = "Naive Bayes Results\n"
            results += "=" * 30 + "\n\n"
            
            # Display feature names
            results += "Features:\n"
            feature_names = getattr(self, 'feature_names', 
                                  [f"Feature {i}" for i in range(self.X_train.shape[1])])
            for i, name in enumerate(feature_names):
                results += f"{i+1}. {name.strip()}\n"
            
            # Display class priors
            results += "\nClass Priors:\n"
            for i, prior in enumerate(model.class_prior_):
                results += f"Class {i}: {prior:.4f}\n"
                
            # Model performance metrics
            results += "\nModel Performance:\n"
            results += f"Accuracy: {accuracy:.4f}\n\n"
            results += "Confusion Matrix:\n"
            results += str(conf_mat) + "\n"
            
            # Display results
            self.metrics_text.setText(results)
            self.visualize_classification_results(model, y_pred)
            
        except Exception as e:
            self.show_error(f"Error displaying Naive Bayes results: {str(e)}")

    def display_svm_results(self, model, y_pred):
        """Display SVM results including metrics and visualization"""
        try:
            # Clear previous results
            self.metrics_text.clear()
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            conf_mat = confusion_matrix(self.y_test, y_pred)
            
            # Format results text
            results = "Support Vector Machine Results\n"
            results += "=" * 30 + "\n\n"
            
            # Support vectors info
            results += f"Number of Support Vectors: {model.n_support_.sum()}\n"
            results += f"Support Vectors per Class: {model.n_support_}\n\n"

            # Filter and display relevant model parameters
            relevant_params = ['gamma', 'coef0', 'class_weight', 'tol', 'max_iter', 'C', 'kernel', 'degree']
            results += "Model Parameters:\n"
            for param in relevant_params:
                if param in model.get_params():
                    results += f"{param}: {model.get_params()[param]}\n"
                
            # Model performance metrics
            results += "\nModel Performance:\n"
            results += f"Accuracy: {accuracy:.4f}\n\n"
            results += "Confusion Matrix:\n"
            results += str(conf_mat) + "\n"
            
            # Display results
            self.metrics_text.setText(results)
            self.visualize_classification_results(model, y_pred)
            
        except Exception as e:
            self.show_error(f"Error displaying SVM results: {str(e)}")

    def display_dt_results(self, model, y_pred):
        """Display Decision Tree results including metrics and visualization"""
        try:
            # Clear previous results
            self.metrics_text.clear()
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            conf_mat = confusion_matrix(self.y_test, y_pred)
            
            # Format results text
            results = "Decision Tree Results\n"
            results += "=" * 30 + "\n\n"
            
            # Model parameters
            results += "Model Parameters:\n"
            relevant_params = ['criterion', 'max_depth', 'min_samples_split', 
                            'min_samples_leaf', 'max_features', 'random_state', 'class_weight']
            for param in relevant_params:
                if param in model.get_params():
                    results += f"{param}: {model.get_params()[param]}\n"
            
            # Model performance metrics
            results += "\nModel Performance:\n"
            results += f"Accuracy: {accuracy:.4f}\n\n"
            results += "Confusion Matrix:\n"
            results += str(conf_mat) + "\n"
            
            # Display results
            self.metrics_text.setText(results)
            self.visualize_classification_results(model, y_pred)
            
        except Exception as e:
            self.show_error(f"Error displaying Decision Tree results: {str(e)}")
    
    def display_rf_results(self, model, y_pred):
        """Display Random Forest results including metrics and visualization"""
        try:
            # Clear previous results
            self.metrics_text.clear()
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            conf_mat = confusion_matrix(self.y_test, y_pred)
            
            # Format results text
            results = "Random Forest Results\n"
            results += "=" * 30 + "\n\n"
            
            # Model parameters
            results += "Model Parameters:\n"
            relevant_params = ['criterion', 'n_estimators', 'max_depth', 'min_samples_split', 
                            'min_samples_leaf', 'max_features', 'bootstrap', 'random_state', 
                            'class_weight', 'max_leaf_nodes', 'min_impurity_decrease']
            for param in relevant_params:
                if param in model.get_params():
                    results += f"{param}: {model.get_params()[param]}\n"
            
            # Model performance metrics
            results += "\nModel Performance:\n"
            results += f"Accuracy: {accuracy:.4f}\n\n"
            results += "Confusion Matrix:\n"
            results += str(conf_mat) + "\n"
            
            # Display results
            self.metrics_text.setText(results)
            self.visualize_classification_results(model, y_pred)
            
        except Exception as e:
            self.show_error(f"Error displaying Random Forest results: {str(e)}")
            
    def display_knn_results(self, model, y_pred):
        """Display KNN results including metrics and visualization"""
        try:
            # Clear previous results
            self.metrics_text.clear()
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            conf_mat = confusion_matrix(self.y_test, y_pred)
            
            # Format results text
            results = "K-Nearest Neighbors Results\n"
            results += "=" * 30 + "\n\n"
            
            # Model parameters
            results += "Model Parameters:\n"
            relevant_params = ['n_neighbors', 'metric', 'weights', 'algorithm', 'leaf_size']
            for param in relevant_params:
                if param in model.get_params():
                    results += f"{param}: {model.get_params()[param]}\n"
            
            # Model performance metrics
            results += "\nModel Performance:\n"
            results += f"Accuracy: {accuracy:.4f}\n\n"
            results += "Confusion Matrix:\n"
            results += str(conf_mat) + "\n"
            
            # Display results
            self.metrics_text.setText(results)
            self.visualize_classification_results(model, y_pred)
            
        except Exception as e:
            self.show_error(f"Error displaying KNN results: {str(e)}")

    def visualize_classification_results(self, model, y_pred):
        """Common visualization method for classification results"""
        try:
            self.figure.clear()
            if self.X_test.shape[1] <= 2:  # 2D visualization
                ax = self.figure.add_subplot(111)
                
                # Create mesh grid for decision boundary
                x_min, x_max = self.X_test[:, 0].min() - 1, self.X_test[:, 0].max() + 1
                y_min, y_max = (self.X_test[:, 1].min() - 1, self.X_test[:, 1].max() + 1) \
                    if self.X_test.shape[1] > 1 else (-1, 1)
                xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                                 np.arange(y_min, y_max, 0.02))
                
                # Plot decision boundary
                try:
                    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
                    Z = Z.reshape(xx.shape)
                    ax.contourf(xx, yy, Z, alpha=0.4)
                except Exception:
                    pass  # Skip decision boundary if it can't be computed
                
                scatter = ax.scatter(self.X_test[:, 0],
                                 self.X_test[:, 1] if self.X_test.shape[1] > 1 \
                                     else np.zeros_like(self.X_test[:, 0]),
                                 c=y_pred,
                                 cmap='viridis')
                ax.set_xlabel("Feature 1")
                ax.set_ylabel("Feature 2" if self.X_test.shape[1] > 1 else "")
                ax.set_title(f"{model.__class__.__name__}: Classification Results")
                self.figure.colorbar(scatter)
            else:  # PCA visualization for high-dimensional data
                pca = PCA(n_components=2)
                X_test_2d = pca.fit_transform(self.X_test)
                ax = self.figure.add_subplot(111)
                scatter = ax.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_pred, cmap='viridis')
                ax.set_xlabel("First Principal Component")
                ax.set_ylabel("Second Principal Component")
                ax.set_title(f"{model.__class__.__name__}: Classification Results (PCA)")
                self.figure.colorbar(scatter)
            
            self.canvas.draw()
            
        except Exception as e:
            self.show_error(f"Error in visualization: {str(e)}")

    def train_model(self, model_name, param_widgets):
        """Train the selected model with parameters"""
        try:
            if self.X_train is None or self.y_train is None:
                raise ValueError("Please load a dataset first")
            
            # Get parameters from widgets
            params = {}
            feature_names = None
            target_name = None
            
            for param_name, widget in param_widgets.items():
                # Skip these parameters as they're not model parameters
                if param_name in ['feature_names', 'target_name', 'test_size']:
                    if param_name == 'feature_names':
                        feature_names = widget.text().strip()
                    elif param_name == 'target_name':
                        target_name = widget.text().strip()
                    continue
                    
                if isinstance(widget, QCheckBox):
                    params[param_name] = widget.isChecked()
                elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    params[param_name] = widget.value()
                elif isinstance(widget, QComboBox):
                    params[param_name] = widget.currentText()
                elif isinstance(widget, QLineEdit):
                    # Handle numeric inputs from QLineEdit
                    try:
                        value = float(widget.text())
                        params[param_name] = value
                    except ValueError:
                        params[param_name] = widget.text()
            
            # Store feature names if provided
            if feature_names:
                self.feature_names = [name.strip() for name in feature_names.split(',')]
            
            # Handle different models
            if model_name == "Linear Regression":
                model = LinearRegression(**params)
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                self.display_linear_regression_results(model, y_pred)
                
            elif model_name == "Logistic Regression":
                model = LogisticRegression(**params)
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                self.display_logistic_regression_results(model, y_pred)

            elif model_name == "Support Vector Regression":
                # Convert numeric parameters
                numeric_params = ['C', 'epsilon']
                for param in numeric_params:
                    if param in params:
                        try:
                            params[param] = float(params[param])
                        except (ValueError, TypeError):
                            params.pop(param)

                # Handle kernel parameter
                if 'kernel' in params:
                    params['kernel'] = params['kernel'].lower()

                # Create and train SVR model
                model = SVR(**params)
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)

                # Display results
                self.display_svr_results(model, y_pred)    

            elif model_name == "Naive Bayes":
                # Handle priors
                priors_type = params.pop('priors_type', 'uniform')
                custom_priors = params.pop('custom_priors', None)
                feature_names = params.pop('feature_names', '').strip()
                target_name = params.pop('target_name', '').strip()

                # Store names for display
                if feature_names:
                    self.feature_names = feature_names.split(',')
                if target_name:
                    self.target_name = target_name
                
                if priors_type == 'custom' and custom_priors:
                    try:
                        # Convert string of comma-separated values to numpy array
                        priors = np.array([float(x.strip()) for x in custom_priors.split(',')])
                        # Normalize priors to sum to 1
                        priors = priors / priors.sum()
                        params['priors'] = priors
                    except Exception as e:
                        self.show_error(f"Error parsing custom priors: {str(e)}")
                        return
                
                model = GaussianNB(**params)
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)

                # Add feature and target names to results
                results = f"Naive Bayes Results\n"
                results += "=" * 30 + "\n\n"
                
                # Display feature names
                results += "Features:\n"
                feature_names = getattr(self, 'feature_names', [f"Feature {i}" for i in range(self.X_train.shape[1])])
                for i, name in enumerate(feature_names):
                    results += f"{i+1}. {name.strip()}\n"
                
                # Display target name
                results += f"\nTarget Variable: {getattr(self, 'target_name', 'Class')}\n\n"
                
                # Display model parameters and metrics
                self.display_naive_bayes_results(model, y_pred)
            
            elif model_name == "Support Vector Machine":
                # Convert string parameters to appropriate types
                if params.get('gamma') in ['scale', 'auto']:
                    gamma = params.get('gamma')
                else:
                    try:
                        gamma = float(params.get('gamma', 'scale'))
                    except ValueError:
                        gamma = 'scale'
                params['gamma'] = gamma

                # Convert numeric parameters
                numeric_params = ['C', 'coef0', 'tol']
                for param in numeric_params:
                    if param in params:
                        try:
                            params[param] = float(params[param])
                        except (ValueError, TypeError):
                            params.pop(param)

                # Convert max_iter to int
                if 'max_iter' in params:
                    try:
                        params['max_iter'] = int(params['max_iter'])
                    except (ValueError, TypeError):
                        params.pop('max_iter')

                # Handle class_weight
                if params.get('class_weight') == 'none':
                    params['class_weight'] = None

                # Create and train SVM model
                model = SVC(**params)
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)

                # Add additional SVM-specific information
                additional_info = "Support Vector Machine Results\n"
                additional_info += "=" * 30 + "\n\n"
                additional_info += f"Number of Support Vectors: {model.n_support_.sum()}\n"
                additional_info += f"Support Vectors per Class: {model.n_support_}\n\n"

                # Use the common classification display method
                self.display_svm_results(model, y_pred)
                
            elif model_name == "Decision Tree":
                # Convert string parameters to appropriate types
                if params.get('max_features') == 'none':
                    params['max_features'] = None
                if params.get('class_weight') == 'none':
                    params['class_weight'] = None

                # Convert numeric parameters
                numeric_params = ['max_depth', 'min_samples_split', 'min_samples_leaf', 'random_state']
                for param in numeric_params:
                    if param in params:
                        try:
                            params[param] = int(params[param])
                        except (ValueError, TypeError):
                            params.pop(param)

                # Create and train Decision Tree model
                model = DecisionTreeClassifier(**params)
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)

                # Display results
                self.display_dt_results(model, y_pred)
                
            elif model_name == "Random Forest":
                # Convert string parameters to appropriate types
                if params.get('max_features') == 'none':
                    params['max_features'] = None
                if params.get('class_weight') == 'none':
                    params['class_weight'] = None

                # Convert numeric parameters
                numeric_params = ['n_estimators', 'max_depth', 'min_samples_split', 
                                  'min_samples_leaf', 'random_state', 'max_leaf_nodes']
                for param in numeric_params:
                    if param in params:
                        try:
                            params[param] = int(params[param])
                        except (ValueError, TypeError):
                            params.pop(param)

                # Convert floating-point parameters
                float_params = ['min_impurity_decrease']
                for param in float_params:
                    if param in params:
                        try:
                            params[param] = float(params[param])
                        except (ValueError, TypeError):
                            params.pop(param)

                # Create and train Random Forest model
                model = RandomForestClassifier(**params)
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)

                # Display results
                self.display_rf_results(model, y_pred)
                
            elif model_name == "K-Nearest Neighbors":
                # Handle custom distance metric
                if params.get('metric') == 'custom':
                    self.show_error("Custom distance functions are not yet supported.")
                    return

                # Convert numeric parameters
                numeric_params = ['n_neighbors', 'p', 'leaf_size']
                for param in numeric_params:
                    if param in params:
                        try:
                            params[param] = int(params[param])
                        except (ValueError, TypeError):
                            params.pop(param)

                # Create and train KNN model
                model = KNeighborsClassifier(**params)
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)

                # Display results
                self.display_knn_results(model, y_pred)
            
            self.status_bar.showMessage(f"Trained {model_name} successfully")
            
        except Exception as e:
            self.show_error(f"Error training model: {str(e)}")
    
    def update_model_details(self, model_params):
        """
        Update the metrics text with model details
        """
        details = "Model Details:\n\n"
        
        # Add coefficients
        if 'coefficients' in model_params:
            details += "Coefficients: "
            if len(model_params['coefficients']) <= 10:
                details += str(model_params['coefficients']) + "\n"
            else:
                details += "[" + ", ".join(f"{c:.4f}" for c in model_params['coefficients'][:5]) + ", ...]\n"
        
        # Add intercept
        if 'intercept' in model_params:
            details += f"Intercept: {model_params['intercept']:.4f}\n"
        
        # Add other parameters
        details += f"Loss Function: {model_params['loss_function']}\n"
        details += f"Fit Intercept: {model_params['fit_intercept']}\n"
        details += f"Normalize: {model_params['normalize']}\n"
        
        # Append to existing metrics
        current_text = self.metrics_text.toPlainText()
        self.metrics_text.setText(current_text + "\n\n" + details)

    def create_algorithm_group(self, name, params):
        """Create a group for algorithm parameters"""
        group = QGroupBox(name)
        layout = QFormLayout()
        
        param_widgets = {}
        
        for param_name, param_type in params.items():
            if param_name in ['feature_names', 'target_name']:
                # Create text input with placeholder
                widget = QLineEdit()
                widget.setPlaceholderText(f"Enter {param_name.replace('_', ' ')} (comma-separated)")
                param_widgets[param_name] = widget
                layout.addRow(f"{param_name.replace('_', ' ').title()}:", widget)
                
            elif isinstance(param_type, list):
                widget = QComboBox()
                widget.addItems([str(item) for item in param_type])
                param_widgets[param_name] = widget
                layout.addRow(f"{param_name.replace('_', ' ').title()}:", widget)
                
            elif param_type == "int":
                widget = QSpinBox()
                widget.setRange(-999999, 999999)
                widget.setValue(0)
                param_widgets[param_name] = widget
                layout.addRow(f"{param_name.replace('_', ' ').title()}:", widget)
                
            elif param_type == "double":
                widget = QDoubleSpinBox()
                widget.setRange(-999999, 999999)
                widget.setValue(0)
                widget.setSingleStep(0.1)
                param_widgets[param_name] = widget
                layout.addRow(f"{param_name.replace('_', ' ').title()}:", widget)
                
            elif param_type == "checkbox":
                widget = QCheckBox()
                widget.setChecked(False)
                param_widgets[param_name] = widget
                layout.addRow(f"{param_name.replace('_', ' ').title()}:", widget)
                
            elif param_type == "text":
                widget = QLineEdit()
                param_widgets[param_name] = widget
                layout.addRow(f"{param_name.replace('_', ' ').title()}:", widget)
        
        # Add train button
        train_btn = QPushButton(f"Train {name}")
        train_btn.clicked.connect(lambda: self.train_model(name, param_widgets))
        layout.addRow(train_btn)
        
        group.setLayout(layout)
        return group

    def show_error(self, message):
        """Show error message dialog"""
        QMessageBox.critical(self, "Error", message)
       
    def create_deep_learning_tab(self):
        """Create the deep learning tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # MLP section
        mlp_group = QGroupBox("Multi-Layer Perceptron")
        mlp_layout = QVBoxLayout()
        
        # Layer configuration
        self.layer_config = []
        layer_btn = QPushButton("Add Layer")
        layer_btn.clicked.connect(self.add_layer_dialog)
        mlp_layout.addWidget(layer_btn)
        
        # Training parameters
        training_params_group = self.create_training_params_group()
        mlp_layout.addWidget(training_params_group)
        
        # Train button
        train_btn = QPushButton("Train Neural Network")
        train_btn.clicked.connect(self.train_neural_network)
        mlp_layout.addWidget(train_btn)
        
        mlp_group.setLayout(mlp_layout)
        layout.addWidget(mlp_group, 0, 0)
        
        # CNN section
        cnn_group = QGroupBox("Convolutional Neural Network")
        cnn_layout = QVBoxLayout()
        
        # CNN architecture controls
        cnn_controls = self.create_cnn_controls()
        cnn_layout.addWidget(cnn_controls)
        
        cnn_group.setLayout(cnn_layout)
        layout.addWidget(cnn_group, 0, 1)
        
        # RNN section
        rnn_group = QGroupBox("Recurrent Neural Network")
        rnn_layout = QVBoxLayout()
        
        # RNN architecture controls
        rnn_controls = self.create_rnn_controls()
        rnn_layout.addWidget(rnn_controls)
        
        rnn_group.setLayout(rnn_layout)
        layout.addWidget(rnn_group, 1, 0)
        
        return widget
    
    def add_layer_dialog(self):
        """Open a dialog to add a neural network layer"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Neural Network Layer")
        layout = QVBoxLayout(dialog)
        
        # Layer type selection
        type_layout = QHBoxLayout()
        type_label = QLabel("Layer Type:")
        type_combo = QComboBox()
        type_combo.addItems(["Dense", "Conv2D", "MaxPooling2D", "Flatten", "Dropout"])
        type_layout.addWidget(type_label)
        type_layout.addWidget(type_combo)
        layout.addLayout(type_layout)
        
        # Parameters input
        params_group = QGroupBox("Layer Parameters")
        params_layout = QVBoxLayout()
        
        # Dynamic parameter inputs based on layer type
        self.layer_param_inputs = {}
        
        def update_params():
            # Clear existing parameter inputs
            for widget in list(self.layer_param_inputs.values()):
                params_layout.removeWidget(widget)
                widget.deleteLater()
            self.layer_param_inputs.clear()
            
            layer_type = type_combo.currentText()
            if layer_type == "Dense":
                units_label = QLabel("Units:")
                units_input = QSpinBox()
                units_input.setRange(1, 1000)
                units_input.setValue(32)
                self.layer_param_inputs["units"] = units_input
                
                activation_label = QLabel("Activation:")
                activation_combo = QComboBox()
                activation_combo.addItems(["relu", "sigmoid", "tanh", "softmax"])
                self.layer_param_inputs["activation"] = activation_combo
                
                params_layout.addWidget(units_label)
                params_layout.addWidget(units_input)
                params_layout.addWidget(activation_label)
                params_layout.addWidget(activation_combo)
            
            elif layer_type == "Conv2D":
                filters_label = QLabel("Filters:")
                filters_input = QSpinBox()
                filters_input.setRange(1, 1000)
                filters_input.setValue(32)
                self.layer_param_inputs["filters"] = filters_input
                
                kernel_label = QLabel("Kernel Size:")
                kernel_input = QLineEdit()
                kernel_input.setText("3, 3")
                self.layer_param_inputs["kernel_size"] = kernel_input
                
                params_layout.addWidget(filters_label)
                params_layout.addWidget(filters_input)
                params_layout.addWidget(kernel_label)
                params_layout.addWidget(kernel_input)
            
            elif layer_type == "Dropout":
                rate_label = QLabel("Dropout Rate:")
                rate_input = QDoubleSpinBox()
                rate_input.setRange(0.0, 1.0)
                rate_input.setValue(0.5)
                rate_input.setSingleStep(0.1)
                self.layer_param_inputs["rate"] = rate_input
                
                params_layout.addWidget(rate_label)
                params_layout.addWidget(rate_input)
        
        type_combo.currentIndexChanged.connect(update_params)
        update_params()  # Initial update
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add Layer")
        cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        def add_layer():
            layer_type = type_combo.currentText()
            
            # Collect parameters
            layer_params = {}
            for param_name, widget in self.layer_param_inputs.items():
                if isinstance(widget, QSpinBox):
                    layer_params[param_name] = widget.value()
                elif isinstance(widget, QDoubleSpinBox):
                    layer_params[param_name] = widget.value()
                elif isinstance(widget, QComboBox):
                    layer_params[param_name] = widget.currentText()
                elif isinstance(widget, QLineEdit):
                    # Handle kernel size or other tuple-like inputs
                    if param_name == "kernel_size":
                        layer_params[param_name] = tuple(map(int, widget.text().split(',')))
            
            self.layer_config.append({
                "type": layer_type,
                "params": layer_params
            })
            
            dialog.accept()
        
        add_btn.clicked.connect(add_layer)
        cancel_btn.clicked.connect(dialog.reject)
        
        dialog.exec()
    
    def create_training_params_group(self):
        """Create group for neural network training parameters"""
        group = QGroupBox("Training Parameters")
        layout = QVBoxLayout()
        
        # Batch size
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Size:"))
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 1000)
        self.batch_size_spin.setValue(32)
        batch_layout.addWidget(self.batch_size_spin)
        layout.addLayout(batch_layout)
        
        # Epochs
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("Epochs:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(10)
        epochs_layout.addWidget(self.epochs_spin)
        layout.addLayout(epochs_layout)
        
        # Learning rate
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("Learning Rate:"))
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setSingleStep(0.001)
        lr_layout.addWidget(self.lr_spin)
        layout.addLayout(lr_layout)
        
        group.setLayout(layout)
        return group
    
    def create_cnn_controls(self):
        """Create controls for Convolutional Neural Network"""
        group = QGroupBox("CNN Architecture")
        layout = QVBoxLayout()
        
        # Placeholder for CNN-specific controls
        label = QLabel("CNN Controls (To be implemented)")
        layout.addWidget(label)
        
        group.setLayout(layout)
        return group
    
    def create_rnn_controls(self):
        """Create controls for Recurrent Neural Network"""
        group = QGroupBox("RNN Architecture")
        layout = QVBoxLayout()
        
        # Placeholder for RNN-specific controls
        label = QLabel("RNN Controls (To be implemented)")
        layout.addWidget(label)
        
        group.setLayout(layout)
        return group
    
    def train_neural_network(self):
        """Train the neural network with current configuration"""
        if not self.layer_config:
            self.show_error("Please add at least one layer to the network")
            return
        
        try:
            # Create and compile model
            model = self.create_neural_network()
            
            # Get training parameters
            batch_size = self.batch_size_spin.value()
            epochs = self.epochs_spin.value()
            learning_rate = self.lr_spin.value()
            
            # Prepare data for neural network
            if len(self.X_train.shape) == 1:
                X_train = self.X_train.reshape(-1, 1)
                X_test = self.X_test.reshape(-1, 1)
            else:
                X_train = self.X_train
                X_test = self.X_test
            
            # One-hot encode target for classification
            y_train = tf.keras.utils.to_categorical(self.y_train)
            y_test = tf.keras.utils.to_categorical(self.y_test)
            
            # Compile model
            optimizer = optimizers.Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer,
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
            
            # Train model
            history = model.fit(X_train, y_train,
                                batch_size=batch_size,
                                epochs=epochs,
                                validation_data=(X_test, y_test),
                                callbacks=[self.create_progress_callback()])
            
            # Update visualization with training history
            self.plot_training_history(history)
            
            self.status_bar.showMessage("Neural Network Training Complete")
            
        except Exception as e:
            self.show_error(f"Error training neural network: {str(e)}")
    
    def create_neural_network(self):
        """Create neural network based on current configuration"""
        model = models.Sequential()
        
        # Add layers based on configuration
        for layer_config in self.layer_config:
            layer_type = layer_config["type"]
            params = layer_config["params"]
            
            if layer_type == "Dense":
                model.add(layers.Dense(**params))
            elif layer_type == "Conv2D":
                # Add input shape for the first layer
                if len(model.layers) == 0:
                    params['input_shape'] = self.X_train.shape[1:]
                model.add(layers.Conv2D(**params))
            elif layer_type == "MaxPooling2D":
                model.add(layers.MaxPooling2D())
            elif layer_type == "Flatten":
                model.add(layers.Flatten())
            elif layer_type == "Dropout":
                model.add(layers.Dropout(**params))
        
        # Add output layer based on number of classes
        num_classes = len(np.unique(self.y_train))
        model.add(layers.Dense(num_classes, activation='softmax'))
                
        return model

   
        
    def train_neural_network(self):
        """Train the neural network"""
        try:
            # Create and compile model
            model = self.create_neural_network()
            
            # Get training parameters
            batch_size = self.batch_size_spin.value()
            epochs = self.epochs_spin.value()
            learning_rate = self.lr_spin.value()
            
            # Compile model
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer,
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
            
            # Train model
            history = model.fit(self.X_train, self.y_train,
                              batch_size=batch_size,
                              epochs=epochs,
                              validation_data=(self.X_test, self.y_test),
                              callbacks=[self.create_progress_callback()])
            
            # Update visualization with training history
            self.plot_training_history(history)
            
        except Exception as e:
            self.show_error(f"Error training neural network: {str(e)}")
            
    def create_progress_callback(self):
        """Create callback for updating progress bar during training"""
        class ProgressCallback(tf.keras.callbacks.Callback):
            def __init__(self, progress_bar):
                super().__init__()
                self.progress_bar = progress_bar
                
            def on_epoch_end(self, epoch, logs=None):
                progress = int(((epoch + 1) / self.params['epochs']) * 100)
                self.progress_bar.setValue(progress)
                
        return ProgressCallback(self.progress_bar)
        
 
    def update_visualization(self, y_pred):
        """Update the visualization with current results"""
        self.figure.clear()
        
        # Create appropriate visualization based on data
        if len(np.unique(self.y_test)) > 10:  # Regression
            ax = self.figure.add_subplot(111)
            ax.scatter(self.y_test, y_pred)
            ax.plot([self.y_test.min(), self.y_test.max()],
                   [self.y_test.min(), self.y_test.max()],
                   'r--', lw=2)
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            
        else:  # Classification
            if self.X_train.shape[1] > 2:  # Use PCA for visualization
                pca = PCA(n_components=2)
                X_test_2d = pca.fit_transform(self.X_test)
                
                ax = self.figure.add_subplot(111)
                scatter = ax.scatter(X_test_2d[:, 0], X_test_2d[:, 1],
                                   c=y_pred, cmap='viridis')
                self.figure.colorbar(scatter)
                
            else:  # Direct 2D visualization
                ax = self.figure.add_subplot(111)
                scatter = ax.scatter(self.X_test[:, 0], self.X_test[:, 1],
                                   c=y_pred, cmap='viridis')
                self.figure.colorbar(scatter)
        
        self.canvas.draw()
        
    def update_metrics(self, y_pred):
        """Update metrics display"""
        metrics_text = "Model Performance Metrics:\n\n"
        
        # Calculate appropriate metrics based on problem type
        if len(np.unique(self.y_test)) > 10:  # Regression
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = self.current_model.score(self.X_test, self.y_test)
            
            metrics_text += f"Mean Squared Error: {mse:.4f}\n"
            metrics_text += f"Root Mean Squared Error: {rmse:.4f}\n"
            metrics_text += f"R² Score: {r2:.4f}"
            
        else:  # Classification
            accuracy = accuracy_score(self.y_test, y_pred)
            conf_matrix = confusion_matrix(self.y_test, y_pred)
            
            metrics_text += f"Accuracy: {accuracy:.4f}\n\n"
            metrics_text += "Confusion Matrix:\n"
            metrics_text += str(conf_matrix)
        
        self.metrics_text.setText(metrics_text)
        
    def plot_training_history(self, history):
        """Plot neural network training history"""
        self.figure.clear()
        
        # Plot training & validation accuracy
        ax1 = self.figure.add_subplot(211)
        ax1.plot(history.history['accuracy'])
        ax1.plot(history.history['val_accuracy'])
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend(['Train', 'Test'])
        
        # Plot training & validation loss
        ax2 = self.figure.add_subplot(212)
        ax2.plot(history.history['loss'])
        ax2.plot(history.history['val_loss'])
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend(['Train', 'Test'])
        
        self.figure.tight_layout()
        self.canvas.draw()
        
    def show_error(self, message):
        """Show error message dialog"""
        QMessageBox.critical(self, "Error", message)

def main():
    """Main function to start the application"""
    app = QApplication(sys.argv)
    window = MLCourseGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
    