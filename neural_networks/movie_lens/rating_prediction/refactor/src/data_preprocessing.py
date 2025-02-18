def clean_data(raw_data):
    """Cleans the raw data by handling missing values and duplicates."""
    # Example implementation
    cleaned_data = raw_data.dropna().drop_duplicates()
    return cleaned_data

def normalize_data(data):
    """Normalizes the data to a standard scale."""
    # Example implementation
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)
    return normalized_data

def split_data(data, target, test_size=0.2, random_state=42):
    """Splits the data into training and testing sets."""
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test