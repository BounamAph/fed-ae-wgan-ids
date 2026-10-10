"""
Classifiers.py
===============
Anomaly Intrusion Detection System (AIDS) Classifiers.

Based on: Arafah et al. (2025), Applied Soft Computing, 168, 112455.

ML Classifiers:
    - Decision Tree (DT)
    - Random Forest (RF)

DL Classifiers:
    - Convolutional Neural Network (CNN)
    - Gated Recurrent Unit (GRU)
    - Long Short-Term Memory (LSTM)
    - Recurrent Neural Network (RNN / SimpleRNN)

Each classifier supports both binary and multiclass classification,
with architectures matching Table 4 of the paper.

Usage:
    from Classifiers import build_all_classifiers, evaluate_classifier
    classifiers = build_all_classifiers(input_dim=24, n_classes=5, dataset='nslkdd')
    results = evaluate_classifier(clf, X_train, y_train, X_test, y_test)
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, 
                             recall_score, f1_score, classification_report,
                             confusion_matrix)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

tf.random.set_seed(42)
np.random.seed(42)


# ============================================================================
# ML CLASSIFIERS
# ============================================================================

def build_decision_tree(n_classes=2, dataset='nslkdd'):
    """
    Build Decision Tree classifier.
    
    Paper config (Table 4):
        NSL-KDD Binary: max_depth=7,  Multi: max_depth=9
        CICIDS  Binary: max_depth=3,  Multi: max_depth=10
    """
    if dataset == 'nslkdd':
        max_depth = 7 if n_classes == 2 else 9
    else:  # cicids
        max_depth = 3 if n_classes == 2 else 10
    
    return DecisionTreeClassifier(max_depth=max_depth, random_state=42)


def build_random_forest(n_classes=2, dataset='nslkdd', n_estimators=100):
    """
    Build Random Forest classifier.
    
    Paper config (Table 4):
        NSL-KDD Binary: max_depth=8,  Multi: max_depth=10
        CICIDS  Binary: max_depth=9,  Multi: max_depth=7
    """
    if dataset == 'nslkdd':
        max_depth = 8 if n_classes == 2 else 10
    else:
        max_depth = 9 if n_classes == 2 else 7
    
    return RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=42, n_jobs=-1
    )


# ============================================================================
# DL CLASSIFIERS
# ============================================================================

def build_cnn(input_dim, n_classes, dataset='nslkdd'):
    """
    Build CNN classifier.
    
    Paper config (Table 4) — NSL-KDD Binary:
        Conv2D(40), kernel=(3,3), ReLU → Dropout(0.1) →
        Dense(40), ReLU → Dense(1), sigmoid
    
    Adapted for 1D input: Conv1D instead of Conv2D.
    Input is reshaped to (features, 1) for Conv1D.
    """
    activation = 'relu' if dataset == 'nslkdd' else 'tanh'
    kernel_size = 3 if dataset == 'nslkdd' else 1
    final_activation = 'sigmoid' if n_classes == 2 else 'softmax'
    output_units = 1 if n_classes == 2 else n_classes
    
    model = Sequential(name='CNN')
    model.add(layers.Input(shape=(input_dim, 1)))
    model.add(layers.Conv1D(40, kernel_size=min(kernel_size, input_dim),
                            activation=activation, padding='same'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Flatten())
    model.add(layers.Dense(40, activation=activation))
    model.add(layers.Dense(output_units, activation=final_activation))
    
    loss = 'binary_crossentropy' if n_classes == 2 else 'sparse_categorical_crossentropy'
    model.compile(optimizer=Adam(learning_rate=1e-3), loss=loss, metrics=['accuracy'])
    
    return model


def build_gru(input_dim, n_classes, dataset='nslkdd'):
    """
    Build GRU classifier.
    
    Paper config (Table 4) — NSL-KDD Binary:
        GRU(30), ReLU, Dropout(0.1) → Dense(10), ReLU →
        Dense(10), ReLU → Dense(1), sigmoid
    """
    activation = 'relu' if dataset == 'nslkdd' else 'tanh'
    final_activation = 'sigmoid' if n_classes == 2 else 'softmax'
    output_units = 1 if n_classes == 2 else n_classes
    gru_units = 30 if dataset == 'nslkdd' else 40
    dense_units = 10 if dataset == 'nslkdd' else 40
    
    model = Sequential(name='GRU')
    model.add(layers.Input(shape=(input_dim, 1)))
    
    if n_classes > 2:
        # Multiclass: stacked GRU
        model.add(layers.GRU(gru_units, activation=activation, return_sequences=True))
        model.add(layers.Dropout(0.1))
        model.add(layers.GRU(gru_units, activation=activation))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(dense_units, activation=activation))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(dense_units, activation=activation))
    else:
        # Binary
        model.add(layers.GRU(gru_units, activation=activation))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(dense_units, activation=activation))
        model.add(layers.Dense(dense_units, activation=activation))
    
    model.add(layers.Dense(output_units, activation=final_activation))
    
    loss = 'binary_crossentropy' if n_classes == 2 else 'sparse_categorical_crossentropy'
    model.compile(optimizer=Adam(learning_rate=1e-3), loss=loss, metrics=['accuracy'])
    
    return model


def build_lstm(input_dim, n_classes, dataset='nslkdd'):
    """
    Build LSTM classifier.
    
    Paper config (Table 4) — NSL-KDD Binary:
        LSTM(10), ReLU → LSTM(20), ReLU, Dropout(0.5) →
        Dense(10), ReLU → Dense(1), sigmoid
    """
    activation = 'relu' if dataset == 'nslkdd' else 'tanh'
    final_activation = 'sigmoid' if n_classes == 2 else 'softmax'
    output_units = 1 if n_classes == 2 else n_classes
    
    model = Sequential(name='LSTM')
    model.add(layers.Input(shape=(input_dim, 1)))
    
    if dataset == 'nslkdd':
        if n_classes > 2:
            model.add(layers.LSTM(10, activation=activation, return_sequences=True))
            model.add(layers.Dropout(0.1))
            model.add(layers.LSTM(20, activation=activation, return_sequences=True))
            model.add(layers.Dropout(0.2))
            model.add(layers.LSTM(20, activation=activation))
            model.add(layers.Dropout(0.1))
        else:
            model.add(layers.LSTM(10, activation=activation, return_sequences=True))
            model.add(layers.LSTM(20, activation=activation))
            model.add(layers.Dropout(0.5))
    else:
        model.add(layers.LSTM(40, activation=activation, return_sequences=True))
        model.add(layers.LSTM(40, activation=activation, return_sequences=True))
        model.add(layers.LSTM(40, activation=activation))
        model.add(layers.Dense(40, activation=activation))
    
    model.add(layers.Dense(10, activation=activation))
    model.add(layers.Dense(output_units, activation=final_activation))
    
    loss = 'binary_crossentropy' if n_classes == 2 else 'sparse_categorical_crossentropy'
    model.compile(optimizer=Adam(learning_rate=1e-3), loss=loss, metrics=['accuracy'])
    
    return model


def build_rnn(input_dim, n_classes, dataset='nslkdd'):
    """
    Build SimpleRNN classifier.
    
    Paper config (Table 4) — NSL-KDD Binary:
        SimpleRNN(30), ReLU, Dropout(0.1) → Dense(10), ReLU →
        Dense(10), ReLU → Dense(1), sigmoid
    """
    activation = 'relu' if dataset == 'nslkdd' else 'tanh'
    final_activation = 'sigmoid' if n_classes == 2 else 'softmax'
    output_units = 1 if n_classes == 2 else n_classes
    rnn_units = 30 if n_classes == 2 else 50
    dropout = 0.1 if n_classes == 2 else 0.2
    
    model = Sequential(name='RNN')
    model.add(layers.Input(shape=(input_dim, 1)))
    
    if n_classes > 2:
        model.add(layers.SimpleRNN(rnn_units, activation=activation, return_sequences=True))
        model.add(layers.Dropout(dropout))
        model.add(layers.SimpleRNN(rnn_units, activation=activation, return_sequences=True))
        model.add(layers.Dropout(dropout))
        model.add(layers.SimpleRNN(rnn_units, activation=activation))
        model.add(layers.Dropout(dropout))
    else:
        model.add(layers.SimpleRNN(rnn_units, activation=activation))
        model.add(layers.Dropout(dropout))
    
    model.add(layers.Dense(10, activation=activation))
    model.add(layers.Dense(10, activation=activation))
    model.add(layers.Dense(output_units, activation=final_activation))
    
    loss = 'binary_crossentropy' if n_classes == 2 else 'sparse_categorical_crossentropy'
    model.compile(optimizer=Adam(learning_rate=1e-3), loss=loss, metrics=['accuracy'])
    
    return model


# ============================================================================
# CLASSIFIER FACTORY
# ============================================================================

def build_all_classifiers(input_dim, n_classes=2, dataset='nslkdd'):
    """
    Build all 6 classifiers as specified in the paper.
    
    Parameters
    ----------
    input_dim : int
        Number of input features (24 for NSL-KDD, 10 for CICIDS).
    n_classes : int
        2 for binary, 5 for NSL-KDD multiclass, 15 for CICIDS multiclass.
    dataset : str
        'nslkdd' or 'cicids' — affects hyperparameters.
    
    Returns
    -------
    classifiers : dict
        Dictionary of {name: classifier_object}.
    """
    classifiers = {
        'DT': build_decision_tree(n_classes, dataset),
        'RF': build_random_forest(n_classes, dataset),
        'CNN': build_cnn(input_dim, n_classes, dataset),
        'GRU': build_gru(input_dim, n_classes, dataset),
        'LSTM': build_lstm(input_dim, n_classes, dataset),
        'RNN': build_rnn(input_dim, n_classes, dataset),
    }
    return classifiers


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def prepare_labels(y_train, y_test, binary=True):
    """
    Encode labels for classification.
    
    Parameters
    ----------
    y_train, y_test : pd.Series
        String labels.
    binary : bool
        If True, map to Normal(0) vs Attack(1).
        If False, encode each category as integer.
    
    Returns
    -------
    y_train_enc, y_test_enc : np.ndarray
        Encoded labels.
    label_encoder : LabelEncoder
        Fitted encoder for inverse transform.
    """
    le = LabelEncoder()
    
    if binary:
        y_tr = y_train.apply(lambda x: 0 if x.lower() == 'normal' else 1)
        y_te = y_test.apply(lambda x: 0 if x.lower() == 'normal' else 1)
        le.fit(['Normal', 'Attack'])
        return y_tr.values, y_te.values, le
    else:
        le.fit(list(set(y_train.unique()) | set(y_test.unique())))
        y_tr = le.transform(y_train)
        y_te = le.transform(y_test)
        return y_tr, y_te, le


def reshape_for_dl(X, model_type='CNN'):
    """
    Reshape 2D data for DL models requiring 3D input.
    
    CNN, GRU, LSTM, RNN all expect shape (samples, timesteps, features).
    We reshape (N, F) → (N, F, 1).
    """
    if len(X.shape) == 2:
        return X.reshape(X.shape[0], X.shape[1], 1)
    return X


def train_classifier(name, clf, X_train, y_train, X_test, y_test,
                     epochs=50, batch_size=32, verbose=1):
    """
    Train a single classifier and return evaluation metrics.
    
    Parameters
    ----------
    name : str
        Classifier name ('DT', 'RF', 'CNN', 'GRU', 'LSTM', 'RNN').
    clf : object
        Classifier (sklearn or keras model).
    X_train, y_train : np.ndarray
        Training data.
    X_test, y_test : np.ndarray
        Testing data.
    epochs : int
        Epochs for DL models.
    batch_size : int
        Batch size for DL models.
    verbose : int
        Verbosity level.
    
    Returns
    -------
    results : dict
        Dictionary with metrics (accuracy, precision, recall, f1).
    """
    is_dl = name in ['CNN', 'GRU', 'LSTM', 'RNN']
    
    if verbose:
        print(f"\n  Training {name}...", end=" ")
    
    if is_dl:
        # Reshape for DL models
        X_tr = reshape_for_dl(X_train)
        X_te = reshape_for_dl(X_test)
        
        early_stop = EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        
        dl_verbose = 0 if verbose < 2 else 1
        clf.fit(
            X_tr, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=dl_verbose
        )
        
        # Predict
        y_pred_proba = clf.predict(X_te, verbose=0)
        
        n_classes = y_pred_proba.shape[-1] if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[-1] > 1 else 2
        
        if n_classes == 1 or (len(y_pred_proba.shape) == 1):
            # Binary with sigmoid
            y_pred = (y_pred_proba.flatten() > 0.5).astype(int)
        else:
            # Multiclass with softmax
            y_pred = np.argmax(y_pred_proba, axis=1)
    else:
        # ML classifiers
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
    
    # Compute metrics
    avg = 'binary' if len(np.unique(y_test)) == 2 else 'weighted'
    
    results = {
        'classifier': name,
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, average=avg, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, average=avg, zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred, average=avg, zero_division=0)),
    }
    
    if verbose:
        print(f"Acc={results['accuracy']:.4f}, "
              f"P={results['precision']:.4f}, "
              f"R={results['recall']:.4f}, "
              f"F1={results['f1_score']:.4f}")
    
    return results, y_pred


def evaluate_all_classifiers(X_train, y_train_cat, X_test, y_test_cat,
                             input_dim, dataset='nslkdd',
                             binary=True, epochs=50, batch_size=32,
                             verbose=1):
    """
    Train and evaluate all 6 classifiers.
    
    Parameters
    ----------
    X_train, X_test : np.ndarray
        Feature matrices.
    y_train_cat, y_test_cat : pd.Series
        Category labels.
    input_dim : int
        Feature dimensionality.
    dataset : str
        'nslkdd' or 'cicids'.
    binary : bool
        Binary or multiclass classification.
    
    Returns
    -------
    all_results : list of dict
        Results for each classifier.
    predictions : dict
        Predictions keyed by classifier name.
    """
    classification_type = "Binary" if binary else "Multiclass"
    
    print(f"\n{'=' * 60}")
    print(f"{classification_type.upper()} CLASSIFICATION — {dataset.upper()}")
    print(f"{'=' * 60}")
    
    # Prepare labels
    y_train_enc, y_test_enc, le = prepare_labels(y_train_cat, y_test_cat, binary)
    
    n_classes = len(np.unique(y_train_enc))
    if binary:
        n_classes = 2
    
    print(f"  Classes: {n_classes}")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Testing samples:  {X_test.shape[0]}")
    
    # Build classifiers
    classifiers = build_all_classifiers(input_dim, n_classes, dataset)
    
    all_results = []
    predictions = {}
    
    for name, clf in classifiers.items():
        results, y_pred = train_classifier(
            name, clf, X_train, y_train_enc, X_test, y_test_enc,
            epochs=epochs, batch_size=batch_size, verbose=verbose
        )
        results['classification'] = classification_type
        all_results.append(results)
        predictions[name] = y_pred
    
    # Summary table
    print(f"\n{'─' * 60}")
    print(f"  {'Classifier':<12} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"{'─' * 60}")
    for r in all_results:
        print(f"  {r['classifier']:<12} {r['accuracy']:>10.4f} {r['precision']:>10.4f} "
              f"{r['recall']:>10.4f} {r['f1_score']:>10.4f}")
    print(f"{'─' * 60}")
    
    return all_results, predictions


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("Classifiers module loaded successfully.")
    print("\nUsage:")
    print("  from Classifiers import evaluate_all_classifiers")
    print("  results, preds = evaluate_all_classifiers(")
    print("      X_train, y_train, X_test, y_test,")
    print("      input_dim=24, dataset='nslkdd', binary=True)")
