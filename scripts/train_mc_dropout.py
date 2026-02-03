import argparse
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import sys
import torch

# Ensure we can import from parent directory
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loaders import DataLoader
from src.data.preprocess import DataPreprocessor
from src.models.mc_dropout import MCDropoutModel
from src.utils.types import CORE_RATIONALES


def train_mc_dropout_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    rationales: list,
    hidden_dims: list = [64, 32],
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
    num_epochs: int = 100,
    batch_size: int = 256,
    num_samples: int = 50,
    weight_decay: float = 1e-4,
    random_seed: int = 21,
    additional_features: list = None,
    save_dir: Path = None,
) -> tuple:
    """Train MC Dropout model for multi-label classification."""
    
    print("=" * 80)
    print("TRAINING MC DROPOUT MODEL")
    print("=" * 80)
    print(f"Rationales: {rationales}")
    print(f"Hidden dims: {hidden_dims}")
    print(f"Dropout rate: {dropout_rate}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print("=" * 80)
    
    # Prepare data
    preprocessor = DataPreprocessor(rationales=rationales)
    
    # Prepare training data
    train_df_clean = train_df.dropna(subset=rationales, how='all').copy()
    print(f"\nTraining samples: {len(train_df_clean)}")
    
    for rationale in rationales:
        n_labeled = train_df_clean[rationale].notna().sum()
        n_positive = (train_df_clean[rationale] == 1).sum()
        if n_labeled > 0:
            print(f"  {rationale}: {n_labeled} labeled ({n_positive} positive, {n_positive/n_labeled*100:.1f}%)")
    
    # Prepare features
    X_train_df = preprocessor.prepare_features(
        train_df_clean, 
        additional_features=additional_features
    )
    X_train_df = preprocessor.encode_categorical(X_train_df, fit=True)
    X_train_df = preprocessor.handle_missing(X_train_df)
    
    # Prepare labels - fill NaN with 0 for multi-label training
    y_train = train_df_clean[rationales].fillna(0).astype(int)
    
    # Scale features
    categorical_encoded_cols = [c for c in X_train_df.columns if c.endswith("_encoded")]
    numerical_cols = [c for c in X_train_df.columns if c not in categorical_encoded_cols]
    
    X_train_scaled = preprocessor.scale_features(X_train_df[numerical_cols], fit=True)
    X_train = np.hstack([X_train_scaled, X_train_df[categorical_encoded_cols].values])
    
    feature_names = list(numerical_cols) + list(categorical_encoded_cols)
    
    # Store the feature columns for validation
    train_feature_cols = X_train_df.columns.tolist()
    
    # Prepare validation data if provided
    X_val, y_val = None, None
    if val_df is not None and len(val_df) > 0:
        val_df_clean = val_df.dropna(subset=rationales, how='all').copy()
        
        X_val_df = preprocessor.prepare_features(
            val_df_clean, 
            # additional_features=additional_features
            use_all_features=True
        )
        X_val_df = preprocessor.encode_categorical(X_val_df, fit=False)
        X_val_df = preprocessor.handle_missing(X_val_df)
        
        # Ensure validation has same columns as training
        # Add missing columns with zeros
        for col in train_feature_cols:
            if col not in X_val_df.columns:
                X_val_df[col] = 0
        
        # Remove extra columns not in training
        X_val_df = X_val_df[train_feature_cols]
        
        y_val = val_df_clean[rationales].fillna(0).astype(int).values
        
        X_val_scaled = preprocessor.scale_features(X_val_df[numerical_cols], fit=False)
        X_val = np.hstack([X_val_scaled, X_val_df[categorical_encoded_cols].values])
        
        print(f"\nValidation samples: {len(X_val)}")
    
    print(f"\nFeature matrix shape: {X_train.shape}")
    print(f"Label matrix shape: {y_train.shape}")
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    model = MCDropoutModel(
        rationales=rationales,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
        num_samples=num_samples,
        weight_decay=weight_decay,
        device=device,
        random_seed=random_seed,
    )
    
    model.feature_names = feature_names
    
    # Train model
    print("\nStarting training...")
    model.fit(
        X_train, 
        y_train.values,
        X_val=X_val,
        y_val=y_val,
        verbose=True
    )
    
    # Evaluate on training data
    print("\n" + "=" * 80)
    print("TRAINING EVALUATION")
    print("=" * 80)
    
    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)
    
    from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
    
    train_metrics = {}
    for i, rationale in enumerate(rationales):
        y_true = y_train.values[:, i]
        y_pred = y_train_pred[:, i]
        y_prob = y_train_prob[:, i]
        
        if y_true.sum() > 0:
            try:
                auc = roc_auc_score(y_true, y_prob)
            except:
                auc = np.nan
            
            train_metrics[rationale] = {
                'accuracy': accuracy_score(y_true, y_pred),
                'f1': f1_score(y_true, y_pred, zero_division=0),
                'auc': auc,
                'n_positive': int(y_true.sum()),
                'positive_rate': float(y_true.mean()),
            }
            
            print(f"\n{rationale}:")
            print(f"  Accuracy: {train_metrics[rationale]['accuracy']:.4f}")
            print(f"  F1 Score: {train_metrics[rationale]['f1']:.4f}")
            print(f"  ROC AUC: {train_metrics[rationale]['auc']:.4f}")
    
    # Evaluate on validation data if available
    val_metrics = {}
    if X_val is not None:
        print("\n" + "=" * 80)
        print("VALIDATION EVALUATION")
        print("=" * 80)
        
        y_val_pred = model.predict(X_val)
        y_val_prob = model.predict_proba(X_val)
        
        for i, rationale in enumerate(rationales):
            y_true = y_val[:, i]
            y_pred = y_val_pred[:, i]
            y_prob = y_val_prob[:, i]
            
            if y_true.sum() > 0:
                try:
                    auc = roc_auc_score(y_true, y_prob)
                except:
                    auc = np.nan
                
                val_metrics[rationale] = {
                    'accuracy': accuracy_score(y_true, y_pred),
                    'f1': f1_score(y_true, y_pred, zero_division=0),
                    'auc': auc,
                    'n_positive': int(y_true.sum()),
                    'positive_rate': float(y_true.mean()),
                }
                
                print(f"\n{rationale}:")
                print(f"  Accuracy: {val_metrics[rationale]['accuracy']:.4f}")
                print(f"  F1 Score: {val_metrics[rationale]['f1']:.4f}")
                print(f"  ROC AUC: {val_metrics[rationale]['auc']:.4f}")
    
    # Save model
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = save_dir / "mc_dropout_model.pkl"
        model.save_model(str(model_path))
        
        preprocessor_path = save_dir / "mc_dropout_preprocessor.pkl"
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(preprocessor, f)
        print(f"\nPreprocessor saved to {preprocessor_path}")
        
        # Save metrics
        metrics_path = save_dir / "mc_dropout_metrics.json"
        metrics = {
            'train': train_metrics,
            'val': val_metrics,
            'hyperparameters': {
                'hidden_dims': hidden_dims,
                'dropout_rate': dropout_rate,
                'learning_rate': learning_rate,
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'num_samples': num_samples,
                'weight_decay': weight_decay,
            },
            'timestamp': datetime.now().isoformat(),
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {metrics_path}")
    
    training_info = {
        'rationales': rationales,
        'n_train': len(X_train),
        'n_val': len(X_val) if X_val is not None else 0,
        'n_features': X_train.shape[1],
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'timestamp': datetime.now().isoformat(),
    }
    
    return model, preprocessor, training_info


def main():
    parser = argparse.ArgumentParser(
        description="Train MC Dropout model for voting rationales"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/Imputing Rationales.csv",
        help="Path to data file",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./models/trained/mc_dropout",
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--rationales",
        nargs="+",
        default=CORE_RATIONALES,
        help="Rationales to train models for",
    )
    parser.add_argument(
        "--hidden_dims",
        nargs="+",
        type=int,
        default=[64, 32],
        help="Hidden layer dimensions",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.2,
        help="Dropout rate (0.1-0.3 recommended)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for Adam",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for training",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of MC samples for prediction",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="L2 regularization strength",
    )
    parser.add_argument(
        "--min_meetings_rat",
        type=int,
        default=1,
        help="Minimum N_Meetings_Rat filter",
    )
    parser.add_argument(
        "--min_dissent",
        type=int,
        default=5,
        help="Minimum N_dissent filter",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Validation set size",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=21,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--additional_features",
        nargs="+",
        default=None,
        help="Additional features to include beyond required features",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MC DROPOUT MODEL TRAINING")
    print("=" * 80)
    print(f"Data path: {args.data_path}")
    print(f"Save directory: {args.save_dir}")
    print(f"Rationales: {args.rationales}")
    print(f"Hidden dims: {args.hidden_dims}")
    print(f"Dropout rate: {args.dropout_rate}")
    print("=" * 80)
    
    # Load data
    loader = DataLoader()
    df = loader.load_data(args.data_path)
    
    # Apply filters
    df = loader.apply_filters(
        min_meetings_rat=args.min_meetings_rat,
        min_dissent=args.min_dissent
    )
    
    # Split data
    train_df, val_df = loader.split_train_test(
        test_size=args.test_size,
        random_seed=args.random_seed,
        label=args.rationales,
    )
    
    print(f"\nTraining set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    
    # Train model
    model, preprocessor, training_info = train_mc_dropout_model(
        train_df=train_df,
        val_df=val_df,
        rationales=args.rationales,
        hidden_dims=args.hidden_dims,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        weight_decay=args.weight_decay,
        random_seed=args.random_seed,
        additional_features=args.additional_features,
        save_dir=args.save_dir,
    )
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()