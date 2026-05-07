import os
import ujson as json
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix



WITH_CHEATER_DIR = r"N:\Desktop\Final Year Project\data\with_cheater_present"
NO_CHEATER_DIR = r"N:\Desktop\Final Year Project\data\no_cheater_present"

CACHE_PATH = "caches_dataset.csv"

def iter_json_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            yield os.path.join(folder_path, filename)


def load_single_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        full = json.load(f)
    
    keys_we_need = [
        "weapon_fire", "player_hurt", "player_death", "bullet_damage",
        "player_jump", "player_footstep",
        "flashbang_detonate", "smokegrenade_detonate",
        "hegrenade_detonate", "inferno_startburn",
        "bomb_planted", "bomb_begindefuse", "bomb_exploded"
    ]

    return {k: full.get(k, []) for k in keys_we_need}

def extract_features(match_json):
    features = {}

    #Combat related features
    features["num_weapon_fires"] = len(match_json.get("weapon_fire", []))
    features["num_hits"] = len(match_json.get("player_hurt", []))
    features["num_kills"] = len(match_json.get("player_death", []))
    features["num_damage_events"] = len(match_json.get("bullet_damage", []))

    #movement related features
    features["num_jumps"] = len(match_json.get("player_jump", []))
    features["num_footsteps"] = len(match_json.get("player_footstep", []))

    #Utility usage
    features["num_flashbangs"] = len(match_json.get("flashbang_detonate", []))
    features["num_smokes"] = len(match_json.get("smokegrenade_detonate", []))
    features["num_hegrenades"] = len(match_json.get("hegrenade_detonate", []))
    features["num_molotovs"] = len(match_json.get("inferno_startburn", []))

    #Bomb interactions
    features["bomb_plants"] = len(match_json.get("bomb_planted", []))
    features["bomb_defuses"] = len(match_json.get("bomb_begindefuse", []))
    features["bomb_explosions"] = len(match_json.get("bomb_exploded", []))

    return features

def build_dataset():
    rows = []

    for file_path in tqdm(list(iter_json_files(WITH_CHEATER_DIR)), desc="Processing with cheater files"):
        match = load_single_json(file_path)
        features = extract_features(match)
        features["label"] = 1
        rows.append(features)

    for file_path in tqdm(list(iter_json_files(NO_CHEATER_DIR)), desc="Processing without cheater files"):
        match = load_single_json(file_path)
        features = extract_features(match)
        features["label"] = 0
        rows.append(features)
    
    return pd.DataFrame(rows)

def preprocess_dataset(df):
    X = df.drop("label", axis=1)
    y = df["label"]

    X_train, X_test,y_train,y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled

def train_models(X_train, y_train, X_train_scaled):
    print("\nRunning Hyperparameter tuning for Decision Tree...")

    dt_params = {
        "max_depth": [5,10,15,None],
        "min_samples_split": [2,5,10],
        "min_samples_leaf": [1,2,4],
        "criterion": ["gini", "entropy"]
    }
    dt_grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        dt_params,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )
    dt_grid.fit(X_train, y_train)
    dt_model = dt_grid.best_estimator_

    print("Best Decision Tree Params:", dt_grid.best_params_)

    print("\nRunning Hyperparamter tuning for SVM...")

    svm_params = {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"]
    }

    svm_grid = GridSearchCV(
        SVC(random_state=42),
        svm_params,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    svm_grid.fit(X_train_scaled, y_train)
    svm_model = svm_grid.best_estimator_

    print("Best SVM Params:", svm_grid.best_params_)

    return dt_model, svm_model

def evaluate_models(dt_model, svm_model, X_test, y_test, X_test_scaled):
    results = {}

    # Decision Tree
    dt_preds = dt_model.predict(X_test)
    results["Decision Tree"] = {
        "accuracy": accuracy_score(y_test, dt_preds),
        "precision": precision_score(y_test, dt_preds),
        "recall": recall_score(y_test, dt_preds),
        "confusion_matrix": confusion_matrix(y_test, dt_preds)
    }

    # SVM
    svm_preds = svm_model.predict(X_test_scaled)
    results["SVM"] = {
        "accuracy": accuracy_score(y_test, svm_preds),
        "precision": precision_score(y_test, svm_preds),
        "recall": recall_score(y_test, svm_preds),
        "confusion_matrix": confusion_matrix(y_test, svm_preds)
    }

    return results


def main():
    if os.path.exists(CACHE_PATH):
        print("\nLoading dataset from cache...")
        df = pd.read_csv(CACHE_PATH)
    else:
        print("\nNo cache found - building dataset from JSON files...")
        df = build_dataset()
        df.to_csv(CACHE_PATH, index=False)
        print("Dataset cached for future runs!")

    print("\nDataset loaded successfully!")
    print(df.head())
    print("\nDataset shape:", df.shape)

    print("\nRunning preprocessing...")
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = preprocess_dataset(df)
    print("Preprocessing complete!")
    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)

    print("\nTraining models...")
    dt_model, svm_model = train_models(X_train, y_train, X_train_scaled)
    print("Models trained successfully!")

    print("\nEvaluating models...")
    results = evaluate_models(dt_model, svm_model, X_test, y_test, X_test_scaled)

    print("\nModel Evaluation Results:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(metric, ":", value)


if __name__ == "__main__":
    main()