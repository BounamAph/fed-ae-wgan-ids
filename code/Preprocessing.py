"""
Preprocessing.py
=================
Data Preprocessing Module for AE-WGAN Anomaly-Based Network Intrusion Detection.

Based on: Arafah et al. (2025), "AE-WGAN: A Novel Approach for Anomaly-Based
           Intrusion Detection Systems", Applied Soft Computing, 168, 112455.

Pipeline Steps:
    1. Data Cleaning     → Remove constant & quasi-constant features
    2. Ordinal Encoding  → Convert categorical features to numeric
    3. Min-Max Scaling   → Normalize all numeric features to [0, 1]
    4. ANOVA Feature Selection → Select top-k features per dataset
       - NSL-KDD:    41 → 24 features
       - CICIDS-2017: 80 → 10 features

Usage:
    from Preprocessing import preprocess_nslkdd, preprocess_cicids
    X_train, X_test, y_train, y_test, feature_names = preprocess_nslkdd(train_path, test_path)
    X_train, X_test, y_train, y_test, feature_names = preprocess_cicids(data_dir)
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# NSL-KDD CONFIGURATION
# ============================================================================

# Column names for NSL-KDD (41 features + label + difficulty)
NSL_KDD_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
    'label', 'difficulty_level'
]

# NSL-KDD attack category mapping
NSL_KDD_ATTACK_MAP = {
    'normal': 'Normal',
    # DoS attacks
    'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS',
    'smurf': 'DoS', 'teardrop': 'DoS', 'mailbomb': 'DoS', 'apache2': 'DoS',
    'processtable': 'DoS', 'udpstorm': 'DoS',
    # Probe attacks
    'satan': 'Probe', 'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe',
    'mscan': 'Probe', 'saint': 'Probe',
    # R2L attacks
    'guess_passwd': 'R2L', 'ftp_write': 'R2L', 'imap': 'R2L', 'phf': 'R2L',
    'multihop': 'R2L', 'warezmaster': 'R2L', 'warezclient': 'R2L', 'spy': 'R2L',
    'xlock': 'R2L', 'xsnoop': 'R2L', 'snmpguess': 'R2L', 'snmpgetattack': 'R2L',
    'httptunnel': 'R2L', 'sendmail': 'R2L', 'named': 'R2L', 'worm': 'R2L',
    # U2R attacks
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'rootkit': 'U2R',
    'perl': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R', 'ps': 'U2R',
    'httptunnel': 'R2L'
}

# Number of features to select via ANOVA
NSL_KDD_N_FEATURES = 24
CICIDS_N_FEATURES = 10


# ============================================================================
# STEP 1: DATA CLEANING
# ============================================================================

def remove_constant_features(df, threshold=0.01):
    """
    Remove constant and quasi-constant features.
    
    Constant features: features with only 1 unique value (variance = 0).
    Quasi-constant features: features where one value dominates 
    (>= 1 - threshold proportion).
    
    Paper reference: "Data cleaning was applied to remove network traffic 
    features that contain fixed values over the entire dataset (constant 
    features) and for features that mostly hold constant values 
    (quasi-constant features)."
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (features only, no labels).
    threshold : float
        Minimum proportion of the dominant value to consider quasi-constant.
    
    Returns
    -------
    df_cleaned : pd.DataFrame
        DataFrame with constant/quasi-constant features removed.
    removed_features : list
        Names of removed features.
    """
    removed = []
    
    for col in df.columns:
        # Constant feature: only 1 unique value
        if df[col].nunique() <= 1:
            removed.append(col)
            continue
        
        # Quasi-constant feature: dominant value >= (1 - threshold)
        dominant_freq = df[col].value_counts(normalize=True).iloc[0]
        if dominant_freq >= (1 - threshold):
            removed.append(col)
    
    df_cleaned = df.drop(columns=removed)
    
    print(f"[Data Cleaning] Removed {len(removed)} constant/quasi-constant features: {removed}")
    
    return df_cleaned, removed


# ============================================================================
# STEP 2: ORDINAL ENCODING
# ============================================================================

def apply_ordinal_encoding(df_train, df_test=None):
    """
    Apply ordinal encoding to convert categorical features to numeric.
    
    Paper reference: "The ordinal encoder was used to convert all 
    non-numeric features into numeric values."
    
    Parameters
    ----------
    df_train : pd.DataFrame
        Training dataframe.
    df_test : pd.DataFrame, optional
        Testing dataframe (encoded with same mapping).
    
    Returns
    -------
    df_train_encoded, df_test_encoded, encoder
    """
    categorical_cols = df_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if len(categorical_cols) == 0:
        print("[Ordinal Encoding] No categorical features found.")
        return df_train, df_test, None
    
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    
    df_train_encoded = df_train.copy()
    df_train_encoded[categorical_cols] = encoder.fit_transform(df_train[categorical_cols])
    
    df_test_encoded = None
    if df_test is not None:
        df_test_encoded = df_test.copy()
        df_test_encoded[categorical_cols] = encoder.transform(df_test[categorical_cols])
    
    print(f"[Ordinal Encoding] Encoded {len(categorical_cols)} categorical features: {categorical_cols}")
    
    return df_train_encoded, df_test_encoded, encoder


# ============================================================================
# STEP 3: MIN-MAX SCALING
# ============================================================================

def apply_minmax_scaling(X_train, X_test=None):
    """
    Apply Min-Max scaling to normalize features to [0, 1].
    
    Paper reference: "Numeric features were scaled through the min-max 
    function, which uniforms all features into the same scale."
    
    Parameters
    ----------
    X_train : np.ndarray or pd.DataFrame
        Training features.
    X_test : np.ndarray or pd.DataFrame, optional
        Testing features.
    
    Returns
    -------
    X_train_scaled, X_test_scaled, scaler
    """
    scaler = MinMaxScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    
    X_test_scaled = None
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
    
    print(f"[Min-Max Scaling] Scaled features to [0, 1] range.")
    
    return X_train_scaled, X_test_scaled, scaler


# ============================================================================
# STEP 4: ANOVA FEATURE SELECTION
# ============================================================================

def apply_anova_selection(X_train, y_train, X_test=None, n_features=24, feature_names=None):
    """
    Apply ANOVA F-test based feature selection.
    
    Paper reference: "Feature selection was conducted after applying 
    feature scaling using ANOVA feature selection to obtain high-quality 
    attacks. ANOVA is a statistical method applied to find the correlation 
    features based on variances between independent features."
    
    NSL-KDD:    41 → 24 features
    CICIDS-2017: 80 → 10 features
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels (encoded).
    X_test : np.ndarray, optional
        Testing features.
    n_features : int
        Number of features to select.
    feature_names : list, optional
        Original feature names for reporting.
    
    Returns
    -------
    X_train_selected, X_test_selected, selector, selected_feature_names
    """
    selector = SelectKBest(score_func=f_classif, k=n_features)
    
    X_train_selected = selector.fit_transform(X_train, y_train)
    
    X_test_selected = None
    if X_test is not None:
        X_test_selected = selector.transform(X_test)
    
    # Get selected feature names
    selected_mask = selector.get_support()
    selected_feature_names = []
    if feature_names is not None:
        selected_feature_names = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
    
    print(f"[ANOVA Selection] Selected {n_features} features from {X_train.shape[1]}.")
    if selected_feature_names:
        print(f"  Selected features: {selected_feature_names}")
    
    return X_train_selected, X_test_selected, selector, selected_feature_names


# ============================================================================
# NSL-KDD FULL PIPELINE
# ============================================================================

def preprocess_nslkdd(train_path, test_path):
    """
    Full preprocessing pipeline for the NSL-KDD dataset.
    
    Steps:
        1. Load data with proper column names
        2. Map attack labels to categories (Normal, DoS, Probe, R2L, U2R)
        3. Data cleaning (remove constant/quasi-constant features)
        4. Ordinal encoding for categorical features
        5. Min-Max scaling
        6. ANOVA feature selection (41 → 24 features)
    
    Parameters
    ----------
    train_path : str
        Path to KDDTrain+.txt or KDDTrain+.csv
    test_path : str
        Path to KDDTest+.txt or KDDTest+.csv
    
    Returns
    -------
    X_train : np.ndarray  — Training features (N_train, 24)
    X_test  : np.ndarray  — Testing features  (N_test, 24)
    y_train : pd.Series   — Training labels
    y_test  : pd.Series   — Testing labels
    selected_features : list — Names of selected features
    preprocessing_objects : dict — Fitted encoder, scaler, selector for reuse
    """
    print("=" * 60)
    print("NSL-KDD PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # --- Load data ---
    df_train = pd.read_csv(train_path, header=None, names=NSL_KDD_COLUMNS)
    df_test = pd.read_csv(test_path, header=None, names=NSL_KDD_COLUMNS)
    
    print(f"\n[Load] Training set: {df_train.shape}")
    print(f"[Load] Testing set:  {df_test.shape}")
    
    # --- Remove difficulty level column ---
    df_train = df_train.drop(columns=['difficulty_level'], errors='ignore')
    df_test = df_test.drop(columns=['difficulty_level'], errors='ignore')
    
    # --- Extract labels ---
    # Remove trailing '.' if present in labels
    df_train['label'] = df_train['label'].str.strip().str.rstrip('.')
    df_test['label'] = df_test['label'].str.strip().str.rstrip('.')
    
    # Map to attack categories
    y_train_cat = df_train['label'].map(NSL_KDD_ATTACK_MAP).fillna('Unknown')
    y_test_cat = df_test['label'].map(NSL_KDD_ATTACK_MAP).fillna('Unknown')
    
    # Keep original detailed labels too
    y_train_detail = df_train['label'].copy()
    y_test_detail = df_test['label'].copy()
    
    # Drop label column from features
    X_train_df = df_train.drop(columns=['label'])
    X_test_df = df_test.drop(columns=['label'])
    
    print(f"\n[Labels] Training distribution (categories):")
    print(y_train_cat.value_counts().to_string())
    print(f"\n[Labels] Testing distribution (categories):")
    print(y_test_cat.value_counts().to_string())
    
    # --- Step 1: Data Cleaning ---
    print(f"\n--- Step 1: Data Cleaning ---")
    # We clean based on training set and apply same removal to test set
    X_train_cleaned, removed_features = remove_constant_features(X_train_df)
    X_test_cleaned = X_test_df.drop(columns=removed_features, errors='ignore')
    
    # --- Step 2: Ordinal Encoding ---
    print(f"\n--- Step 2: Ordinal Encoding ---")
    X_train_encoded, X_test_encoded, encoder = apply_ordinal_encoding(
        X_train_cleaned, X_test_cleaned
    )
    
    feature_names_after_encoding = X_train_encoded.columns.tolist()
    
    # --- Step 3: Min-Max Scaling ---
    print(f"\n--- Step 3: Min-Max Scaling ---")
    X_train_scaled, X_test_scaled, scaler = apply_minmax_scaling(
        X_train_encoded.values, X_test_encoded.values
    )
    
    # --- Step 4: ANOVA Feature Selection ---
    print(f"\n--- Step 4: ANOVA Feature Selection ---")
    # Encode labels for ANOVA (needs numeric labels)
    label_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    y_train_encoded = label_encoder.fit_transform(
        y_train_cat.values.reshape(-1, 1)
    ).ravel()
    y_test_encoded = label_encoder.transform(
        y_test_cat.values.reshape(-1, 1)
    ).ravel()
    
    X_train_final, X_test_final, selector, selected_features = apply_anova_selection(
        X_train_scaled, y_train_encoded,
        X_test_scaled,
        n_features=NSL_KDD_N_FEATURES,
        feature_names=feature_names_after_encoding
    )
    
    print(f"\n[Final] Training features shape: {X_train_final.shape}")
    print(f"[Final] Testing features shape:  {X_test_final.shape}")
    print("=" * 60)
    
    preprocessing_objects = {
        'encoder': encoder,
        'scaler': scaler,
        'selector': selector,
        'label_encoder': label_encoder,
        'removed_features': removed_features,
        'selected_features': selected_features,
    }
    
    return (X_train_final, X_test_final,
            y_train_cat, y_test_cat,
            y_train_detail, y_test_detail,
            selected_features, preprocessing_objects)


# ============================================================================
# CICIDS-2017 FULL PIPELINE
# ============================================================================

def preprocess_cicids(data_dir, test_size=0.2, random_state=42):
    """
    Full preprocessing pipeline for the CICIDS-2017 dataset.
    
    Steps:
        1. Load and merge all CSV files from the data directory
        2. Handle missing/infinite values
        3. Data cleaning (remove constant/quasi-constant features)
        4. Ordinal encoding (if applicable)
        5. Min-Max scaling
        6. ANOVA feature selection (80 → 10 features)
    
    Parameters
    ----------
    data_dir : str
        Path to directory containing CICIDS-2017 CSV files.
    test_size : float
        Proportion for test split.
    random_state : int
        Random seed for reproducibility.
    
    Returns
    -------
    X_train, X_test, y_train, y_test, selected_features, preprocessing_objects
    """
    print("=" * 60)
    print("CICIDS-2017 PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # --- Load data ---
    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
    print(f"\n[Load] Found {len(csv_files)} CSV files in {data_dir}")
    
    dfs = []
    for f in csv_files:
        filepath = os.path.join(data_dir, f)
        df = pd.read_csv(filepath, low_memory=False)
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        dfs.append(df)
        print(f"  Loaded {f}: {df.shape}")
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"\n[Load] Combined dataset: {df.shape}")
    
    # --- Identify label column ---
    label_col = None
    for candidate in ['Label', 'label', ' Label']:
        if candidate in df.columns:
            label_col = candidate
            break
    
    if label_col is None:
        raise ValueError("Could not find label column in CICIDS-2017 data.")
    
    # --- Handle missing / infinite values ---
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    print(f"[Clean] After removing NaN/Inf: {df.shape}")
    
    # --- Extract labels ---
    y = df[label_col].str.strip()
    X_df = df.drop(columns=[label_col])
    
    # Convert all columns to numeric where possible
    for col in X_df.columns:
        X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
    X_df = X_df.dropna(axis=1, how='all')  # drop fully non-numeric cols
    
    print(f"\n[Labels] Distribution:")
    print(y.value_counts().to_string())
    
    # Create binary label (BENIGN vs Attack)
    y_binary = y.apply(lambda x: 'Normal' if x == 'BENIGN' else 'Attack')
    
    # --- Train/Test split (stratified) ---
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=test_size, random_state=random_state, stratify=y
    )
    y_train_binary = y_binary.loc[X_train_df.index]
    y_test_binary = y_binary.loc[X_test_df.index]
    
    X_train_df = X_train_df.reset_index(drop=True)
    X_test_df = X_test_df.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    y_train_binary = y_train_binary.reset_index(drop=True)
    y_test_binary = y_test_binary.reset_index(drop=True)
    
    print(f"\n[Split] Train: {X_train_df.shape}, Test: {X_test_df.shape}")
    
    # --- Step 1: Data Cleaning ---
    print(f"\n--- Step 1: Data Cleaning ---")
    X_train_cleaned, removed_features = remove_constant_features(X_train_df)
    X_test_cleaned = X_test_df.drop(columns=removed_features, errors='ignore')
    
    # --- Step 2: Ordinal Encoding ---
    print(f"\n--- Step 2: Ordinal Encoding ---")
    X_train_encoded, X_test_encoded, encoder = apply_ordinal_encoding(
        X_train_cleaned, X_test_cleaned
    )
    
    feature_names_after_encoding = X_train_encoded.columns.tolist()
    
    # --- Step 3: Min-Max Scaling ---
    print(f"\n--- Step 3: Min-Max Scaling ---")
    X_train_scaled, X_test_scaled, scaler = apply_minmax_scaling(
        X_train_encoded.values.astype(np.float64),
        X_test_encoded.values.astype(np.float64)
    )
    
    # --- Step 4: ANOVA Feature Selection ---
    print(f"\n--- Step 4: ANOVA Feature Selection ---")
    label_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    y_train_enc = label_enc.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    
    X_train_final, X_test_final, selector, selected_features = apply_anova_selection(
        X_train_scaled, y_train_enc,
        X_test_scaled,
        n_features=CICIDS_N_FEATURES,
        feature_names=feature_names_after_encoding
    )
    
    print(f"\n[Final] Training features shape: {X_train_final.shape}")
    print(f"[Final] Testing features shape:  {X_test_final.shape}")
    print("=" * 60)
    
    preprocessing_objects = {
        'encoder': encoder,
        'scaler': scaler,
        'selector': selector,
        'label_encoder': label_enc,
        'removed_features': removed_features,
        'selected_features': selected_features,
    }
    
    return (X_train_final, X_test_final,
            y_train, y_test,
            y_train_binary, y_test_binary,
            selected_features, preprocessing_objects)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_attack_data(X, y_detail, attack_type):
    """
    Extract samples for a specific attack type.
    Used for per-attack AE-WGAN training.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y_detail : pd.Series
        Detailed attack labels (e.g., 'neptune', 'smurf', ...).
    attack_type : str
        Specific attack name to extract.
    
    Returns
    -------
    X_attack : np.ndarray
        Subset of X matching the attack type.
    """
    mask = y_detail == attack_type
    return X[mask]


def get_category_data(X, y_cat, category):
    """
    Extract samples for a specific attack category.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y_cat : pd.Series
        Category labels (Normal, DoS, Probe, R2L, U2R).
    category : str
        Category to extract.
    
    Returns
    -------
    X_category : np.ndarray
    """
    mask = y_cat == category
    return X[mask]


def print_dataset_summary(X_train, X_test, y_train_cat, y_test_cat):
    """Print a summary of the preprocessed dataset."""
    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples:  {X_test.shape[0]}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"\nTraining label distribution:")
    print(y_train_cat.value_counts().to_string())
    print(f"\nTesting label distribution:")
    print(y_test_cat.value_counts().to_string())
    print("=" * 50)


# ============================================================================
# MAIN (demonstration / standalone test)
# ============================================================================

if __name__ == "__main__":
    print("Preprocessing module loaded successfully.")
    print("Usage:")
    print("  from Preprocessing import preprocess_nslkdd, preprocess_cicids")
    print()
    print("  # NSL-KDD:")
    print("  results = preprocess_nslkdd('KDDTrain+.txt', 'KDDTest+.txt')")
    print()
    print("  # CICIDS-2017:")
    print("  results = preprocess_cicids('./CICIDS-2017/')")
