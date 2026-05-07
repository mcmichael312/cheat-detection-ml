# cheat-detection-ml
Machine-learning-based behaviour cheat detection in online multiplayer games.

A final‑year project investigating whether machine‑learning models can detect cheating behaviour in online multiplayer games using gameplay event data.


📌 Project Summary
This project builds a full machine‑learning pipeline that uses Counter‑Strike 2 match telemetry to classify whether a match contains a cheater. The system extracts behavioural features from JSON logs, constructs a labelled dataset, preprocesses the data, trains two supervised learning models (Decision Tree and SVM), and evaluates their performance.

The goal is to explore whether behavioural data alone can support cheat detection without intrusive client‑side anti‑cheat tools.

🧠 Features Extracted
The script extracts counts of key gameplay events, including:

Weapon fires

Hits, kills, and damage events

Jumps and footsteps

Flashbangs, smokes, HE grenades, molotovs

Bomb plants, defuses, and explosions

These features form the input to the machine‑learning models.

⚙️ How the System Works
1. Load JSON Files
Two folders are processed:

with_cheater_present → label 1

no_cheater_present → label 0

2. Extract Features
Only required event categories are loaded to reduce overhead.

3. Build Dataset
All extracted features are combined into a Pandas DataFrame.

4. Preprocessing
Train/test split

Standard scaling for SVM

5. Model Training
Hyperparameter tuning is performed using GridSearchCV for:

Decision Tree

Support Vector Machine (SVM)

6. Evaluation
Both models are evaluated using:

Accuracy

Precision

Recall

Confusion matrix
📊 Results Summary
After tuning, the SVM achieved the strongest overall performance, showing that behavioural features can be effective for cheat detection. The Decision Tree also performed well and provides interpretability.

📘 Dataset Source
This project uses publicly available Counter‑Strike 2 telemetry data:

Gianna, G. & Cayla, C. (2023). ds340w_GiannaCayla – Counter‑Strike 2 Telemetry Dataset. GitHub.  
https://github.com/gkd5216/ds340w_GiannaCayla
