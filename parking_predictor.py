import json
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os

class SmartParkingPredictor:
    def __init__(self, history_file='history.json', model_file='parking_model.pkl'):
        # Use absolute paths to ensure files are found regardless of working directory
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.history_file = os.path.join(script_dir, history_file) if not os.path.isabs(history_file) else history_file
        self.model_file = os.path.join(script_dir, model_file) if not os.path.isabs(model_file) else model_file
        self.model = None
        self.label_encoders = {}
        self.column_mapping = {}
        
    def standardize_columns(self, df):
        """Standardize column names to handle different formats"""
        # Print original columns for debugging (commented out for production)
        # print(f"Original columns: {df.columns.tolist()}", file=sys.stderr)
        
        # Create lowercase mapping for case-insensitive matching
        col_lower = {col.lower(): col for col in df.columns}
        
        # Define possible column name variations
        column_variations = {
            'parking_spot_id': ['Parking_Spot_ID', 'parkingspotid', 'spot_id', 'spotid', 'parking_id', 'id'],
            'Occupancy_Status': ['Occupancy_Status', 'occupancystatus', 'occupancy', 'status', 'occupied'],
            'timestamp': ['Timestamp', 'time', 'datetime', 'date', 'recorded_time'],
            'location': ['Location', 'zone', 'area', 'region'],
            'parking_type': ['Parking_Type', 'parkingtype', 'type', 'spot_type']
        }
        
        # Map columns
        for standard_name, variations in column_variations.items():
            for var in variations:
                if var in col_lower:
                    original_col = col_lower[var]
                    if original_col != standard_name:
                        df = df.rename(columns={original_col: standard_name})
                        self.column_mapping[standard_name] = original_col
                    break
        
        # print(f"Standardized columns: {df.columns.tolist()}", file=sys.stderr)
        return df
    
    def convert_occupancy_to_numeric(self, df):
        """Convert occupancy status to numeric (0=vacant, 1=occupied)"""
        if 'Occupancy_Status' not in df.columns:
            raise ValueError("No Occupancy_Status column found after standardization")
        
        # Check if already numeric
        if pd.api.types.is_numeric_dtype(df['Occupancy_Status']):
            # Ensure it's 0 or 1
            df['Occupancy_Status'] = df['Occupancy_Status'].astype(int)
            df['Occupancy_Status'] = df['Occupancy_Status'].apply(lambda x: 1 if x > 0 else 0)
        else:
            # Convert string values to numeric
            # Handle various string formats
            occupancy_map = {
                'occupied': 1, 'Occupied': 1, 'OCCUPIED': 1, 'yes': 1, 'Yes': 1, 'YES': 1,
                'true': 1, 'True': 1, 'TRUE': 1, '1': 1, 1: 1,
                'vacant': 0, 'Vacant': 0, 'VACANT': 0, 'no': 0, 'No': 0, 'NO': 0,
                'false': 0, 'False': 0, 'FALSE': 0, 'empty': 0, 'Empty': 0, '0': 0, 0: 0,
                'available': 0, 'Available': 0, 'AVAILABLE': 0
            }
            
            df['Occupancy_Status'] = df['Occupancy_Status'].map(occupancy_map)
            
            # Handle any unmapped values
            if df['Occupancy_Status'].isna().any():
                print(f"Warning: Found unmapped occupancy values. Setting to 0 (vacant)", file=sys.stderr)
                df['Occupancy_Status'] = df['Occupancy_Status'].fillna(0)
        
        # Convert to numeric explicitly
        df['Occupancy_Status'] = pd.to_numeric(df['Occupancy_Status'], errors='coerce')
        df['Occupancy_Status'] = df['Occupancy_Status'].fillna(0).astype(int)
        
        # print(f"Occupancy values: {df['Occupancy_Status'].unique()}", file=sys.stderr)
        return df
    
    def create_temporal_features(self, df):
        """Extract temporal features from timestamp"""
        # Find and standardize timestamp column
        timestamp_col = 'timestamp'
        if timestamp_col not in df.columns:
            # Try to find any date/time column
            for col in df.columns:
                if 'time' in col.lower() or 'date' in col.lower():
                    df = df.rename(columns={col: 'timestamp'})
                    break
        
        if 'timestamp' not in df.columns:
            # Create synthetic timestamps if none exist
            print("Warning: No timestamp column found. Creating synthetic timestamps.", file=sys.stderr)
            df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='H')
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Handle any NaT values
        if df['timestamp'].isna().any():
            print("Warning: Some timestamps couldn't be parsed. Filling with synthetic dates.", file=sys.stderr)
            df['timestamp'] = df['timestamp'].fillna(method='ffill').fillna(pd.Timestamp.now())
        
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_peak_hour'] = ((df['hour'] >= 8) & (df['hour'] <= 10) | 
                              (df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
        return df
    
    def create_lag_features(self, df):
        """Create lag features for time series prediction"""
        # Ensure we have a parking_spot_id column
        if 'parking_spot_id' not in df.columns:
            # Check if there's any ID column
            id_cols = [col for col in df.columns if 'id' in col.lower() or 'spot' in col.lower()]
            if id_cols:
                df = df.rename(columns={id_cols[0]: 'parking_spot_id'})
            else:
                # Create synthetic IDs
                print("Warning: No parking_spot_id found. Creating synthetic IDs.", file=sys.stderr)
                df['parking_spot_id'] = 'SPOT_' + (df.index // 24).astype(str)
        
        df = df.sort_values('timestamp')
        
        # Create lag features with explicit numeric conversion
        for lag in [1, 2, 3, 6, 12, 24]:
            df[f'occupancy_lag_{lag}'] = df.groupby('parking_spot_id')['Occupancy_Status'].shift(lag)
            # Fill NaN with 0 and ensure numeric type
            df[f'occupancy_lag_{lag}'] = pd.to_numeric(df[f'occupancy_lag_{lag}'], errors='coerce').fillna(0)
        
        # Rolling statistics with explicit numeric conversion
        for window in [3, 6, 12]:
            df[f'occupancy_rolling_mean_{window}'] = (
                df.groupby('parking_spot_id')['Occupancy_Status']
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            )
            # Fill NaN and ensure numeric
            df[f'occupancy_rolling_mean_{window}'] = pd.to_numeric(
                df[f'occupancy_rolling_mean_{window}'], errors='coerce'
            ).fillna(0)
        
        return df
    
    def train_model(self, data_path=None):
        """Train the parking occupancy prediction model"""
        try:
            # Load data
            if data_path and os.path.exists(data_path):
                print(f"Loading training data from: {data_path}", file=sys.stderr)
                df = pd.read_csv(data_path)
            elif os.path.exists(self.history_file):
                print(f"Loading training data from history: {self.history_file}", file=sys.stderr)
                df = self.load_history()
            else:
                return {"error": "No training data available"}
            
            if df.empty:
                return {"error": "Dataset is empty"}
            
            print(f"Initial dataset shape: {df.shape}", file=sys.stderr)
            
            # Standardize column names
            df = self.standardize_columns(df)
            
            # Convert occupancy to numeric
            df = self.convert_occupancy_to_numeric(df)
            
            # Create features
            df = self.create_temporal_features(df)
            df = self.create_lag_features(df)
            
            # Remove rows with NaN from lag features (only initial rows)
            initial_rows = len(df)
            df = df.dropna(subset=['Occupancy_Status'])
            print(f"Removed {initial_rows - len(df)} rows with missing occupancy", file=sys.stderr)
            
            if len(df) < 50:
                return {"error": f"Insufficient data for training (have {len(df)}, need at least 50 records)"}
            
            # Encode categorical variables
            categorical_cols = ['parking_spot_id', 'location', 'parking_type']
            for col in categorical_cols:
                if col in df.columns:
                    le = LabelEncoder()
                    # Convert to string and handle missing values
                    df[col] = df[col].astype(str).fillna('unknown')
                    df[col + '_encoded'] = le.fit_transform(df[col])
                    self.label_encoders[col] = le
            
            # Prepare features and target
            feature_cols = ['hour', 'day_of_week', 'day_of_month', 'month', 
                           'is_weekend', 'is_peak_hour']
            
            # Add encoded categorical features
            for col in categorical_cols:
                if col + '_encoded' in df.columns:
                    feature_cols.append(col + '_encoded')
            
            # Add lag features
            lag_cols = [col for col in df.columns if 'lag' in col or 'rolling' in col]
            feature_cols.extend(lag_cols)
            
            # Ensure all feature columns exist and are numeric
            for col in feature_cols:
                if col not in df.columns:
                    df[col] = 0
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            X = df[feature_cols]
            y = df['Occupancy_Status']
            
            print(f"Feature matrix shape: {X.shape}", file=sys.stderr)
            print(f"Features used: {feature_cols}", file=sys.stderr)
            print(f"Target distribution: {y.value_counts().to_dict()}", file=sys.stderr)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train model with Gradient Boosting
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            self.model.fit(X_train, y_train)
            
            # Calculate accuracy
            train_acc = self.model.score(X_train, y_train)
            test_acc = self.model.score(X_test, y_test)
            
            # Feature importance
            feature_importance = dict(zip(feature_cols, self.model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Save model
            model_data = {
                'model': self.model,
                'label_encoders': self.label_encoders,
                'feature_cols': feature_cols,
                'column_mapping': self.column_mapping
            }
            with open(self.model_file, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"Model trained successfully!", file=sys.stderr)
            
            return {
                "status": "success",
                "train_accuracy": round(float(train_acc), 4),
                "test_accuracy": round(float(test_acc), 4),
                "features_used": len(feature_cols),
                "training_samples": int(len(X_train)),
                "test_samples": int(len(X_test)),
                "top_features": [{"feature": f, "importance": round(float(imp), 4)} 
                                for f, imp in top_features]
            }
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error during training: {error_trace}", file=sys.stderr)
            return {"error": str(e), "traceback": error_trace}
    
    def load_history(self):
        """Load and parse history.json file"""
        try:
            with open(self.history_file, 'r') as f:
                data = json.load(f)
            
            # Convert to DataFrame
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame([data])
            
            return df
        except Exception as e:
            print(f"Error loading history: {e}", file=sys.stderr)
            return pd.DataFrame()
    
    def record_history(self, spot_data):
        """Record new parking spot data to history.json"""
        history = []
        
        # Load existing history
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    history = json.load(f)
                    if not isinstance(history, list):
                        history = [history]
            except Exception as e:
                print(f"Error loading history: {e}", file=sys.stderr)
                history = []
        
        # Add timestamp if not present
        if 'timestamp' not in spot_data:
            spot_data['timestamp'] = datetime.now().isoformat()
        
        # Normalize Occupancy_Status
        if 'Occupancy_Status' in spot_data:
            if isinstance(spot_data['Occupancy_Status'], str):
                occupancy_map = {
                    'occupied': 1, 'Occupied': 1, 'OCCUPIED': 1,
                    'vacant': 0, 'Vacant': 0, 'VACANT': 0, 'available': 0
                }
                spot_data['Occupancy_Status'] = occupancy_map.get(spot_data['Occupancy_Status'], 0)
        
        # Append new data
        history.append(spot_data)
        
        # Keep only last 10000 records
        if len(history) > 10000:
            history = history[-10000:]
        
        # Save updated history
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        return {"status": "recorded", "total_records": len(history), "latest": spot_data}
    
    def predict_occupancy(self, input_data):
        """Predict occupancy for given parking spot and time"""
        try:
            # Load model if not loaded
            if self.model is None:
                # print(f"Looking for model at: {self.model_file}", file=sys.stderr)
                # print(f"Model file exists: {os.path.exists(self.model_file)}", file=sys.stderr)
                if not os.path.exists(self.model_file):
                    return {"error": f"Model not trained yet. Train the model first. Looking at: {self.model_file}"}
                
                with open(self.model_file, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data['model']
                    self.label_encoders = model_data['label_encoders']
                    feature_cols = model_data['feature_cols']
                    if 'column_mapping' in model_data:
                        self.column_mapping = model_data['column_mapping']
            else:
                with open(self.model_file, 'rb') as f:
                    model_data = pickle.load(f)
                    feature_cols = model_data['feature_cols']
            
            # Parse input timestamp
            if 'timestamp' in input_data:
                ts = pd.to_datetime(input_data['timestamp'])
            else:
                ts = datetime.now()
            
            # Create feature dictionary
            features = {
                'hour': ts.hour,
                'day_of_week': ts.weekday(),
                'day_of_month': ts.day,
                'month': ts.month,
                'is_weekend': int(ts.weekday() >= 5),
                'is_peak_hour': int((ts.hour >= 8 and ts.hour <= 10) or 
                                   (ts.hour >= 17 and ts.hour <= 19))
            }
            
            # Encode categorical variables
            for col in ['parking_spot_id', 'location', 'parking_type']:
                if col in input_data and col in self.label_encoders:
                    try:
                        features[col + '_encoded'] = self.label_encoders[col].transform(
                            [str(input_data[col])]
                        )[0]
                    except:
                        # Unknown category, use default
                        features[col + '_encoded'] = 0
                else:
                    features[col + '_encoded'] = 0
            
            # Get lag features from history
            history_df = self.load_history()
            if not history_df.empty and 'parking_spot_id' in input_data:
                # Standardize and convert history
                history_df = self.standardize_columns(history_df)
                history_df = self.convert_occupancy_to_numeric(history_df)
                
                # Determine timestamp column name (handle both 'timestamp' and 'Timestamp')
                ts_col = 'timestamp' if 'timestamp' in history_df.columns else 'Timestamp'
                
                # Check if parking_spot_id exists after standardization
                if 'parking_spot_id' in history_df.columns:
                    spot_history = history_df[
                        history_df['parking_spot_id'] == input_data['parking_spot_id']
                    ].sort_values(ts_col, ascending=False)
                elif 'Parking_Spot_ID' in history_df.columns:
                    spot_history = history_df[
                        history_df['Parking_Spot_ID'] == input_data['parking_spot_id']
                    ].sort_values(ts_col, ascending=False)
                else:
                    # Use all history if no parking_spot_id column
                    spot_history = history_df.sort_values(ts_col, ascending=False)
                
                if not spot_history.empty:
                    for i, lag in enumerate([1, 2, 3, 6, 12, 24]):
                        if i < len(spot_history):
                            features[f'occupancy_lag_{lag}'] = int(spot_history.iloc[i]['Occupancy_Status'])
                        else:
                            features[f'occupancy_lag_{lag}'] = 0
                    
                    # Calculate rolling means
                    recent_occupancy = spot_history['Occupancy_Status'].head(24).values
                    for window in [3, 6, 12]:
                        if len(recent_occupancy) >= window:
                            features[f'occupancy_rolling_mean_{window}'] = float(np.mean(
                                recent_occupancy[:window]
                            ))
                        else:
                            features[f'occupancy_rolling_mean_{window}'] = 0.0
                else:
                    self._set_default_lag_features(features)
            else:
                self._set_default_lag_features(features)
            
            # Create feature vector in correct order
            X = pd.DataFrame([features])[feature_cols]
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X)[0]
            
            return {
                "parking_spot_id": input_data.get('parking_spot_id', 'unknown'),
                "predicted_occupancy": int(prediction),
                "prediction_label": "Occupied" if prediction == 1 else "Vacant",
                "probability_vacant": round(float(probability[0]), 4),
                "probability_occupied": round(float(probability[1]), 4),
                "prediction_time": ts.isoformat(),
                "confidence": round(float(max(probability)), 4)
            }
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error during prediction: {error_trace}", file=sys.stderr)
            return {"error": str(e), "traceback": error_trace}
    
    def _set_default_lag_features(self, features):
        """Set default values for lag features"""
        for lag in [1, 2, 3, 6, 12, 24]:
            features[f'occupancy_lag_{lag}'] = 0
        for window in [3, 6, 12]:
            features[f'occupancy_rolling_mean_{window}'] = 0.0

def main():
    """Main function for Node-RED exec node integration"""
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No command specified. Use: train, predict, or record"}))
        return
    
    command = sys.argv[1]
    predictor = SmartParkingPredictor()
    
    if command == "train":
        # Train model with optional dataset path
        dataset_path = sys.argv[2] if len(sys.argv) > 2 else None
        result = predictor.train_model(dataset_path)
        print(json.dumps(result))
    
    elif command == 'predict':
        if len(sys.argv) >= 3:
            inputdata = json.loads(sys.argv[2])
        else:
            try:
                # Try to get JSON dict from stdin (Node-RED sends payload here)
                input_str = sys.stdin.read()
                if input_str.strip():
                    inputdata = json.loads(input_str)
                else:
                    # Create default input data for prediction
                    inputdata = {
                        "parking_spot_id": "smart_parking_lot",
                        "timestamp": datetime.now().isoformat(),
                        "location": "main_lot",
                        "parking_type": "general"
                    }
            except Exception as e:
                print(json.dumps({"error": f"No input data provided or invalid JSON via stdin: {e}"}))
                return
        
        result = predictor.predict_occupancy(inputdata)
        
        # Return the result directly as JSON
        print(json.dumps(result))

    
    elif command == "record":
        # Record new data to history
        if len(sys.argv) < 3:
            print(json.dumps({"error": "No data to record"}))
            return
        
        spot_data = json.loads(sys.argv[2])
        result = predictor.record_history(spot_data)
        print(json.dumps(result))
    
    else:
        print(json.dumps({"error": f"Unknown command: {command}"}))

if __name__ == "__main__":
    main()