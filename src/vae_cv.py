import optuna
from sklearn.model_selection import cross_val_score, KFold
from vae import VAE, criterion, load_data, data_to_mat
import torch
import numpy as np
from tqdm import tqdm
from constatants import NS
import pickle as pk

def objective(trial):
    # ハイパーパラメータのサンプル
    z_dim = trial.suggest_int('z_dim', 2, 5)
    lr_vae = trial.suggest_float('lr_vae', 1e-5, 2e-2)
    h_dim = trial.suggest_int('h_dim', 16, 64)
    epoch = trial.suggest_int('epoch', 30, 150)
    monte = trial.suggest_int('monte', 2, 6)
    
    # モデル作成
    model = VAE(z_dim=z_dim, lr=lr_vae, h_dim=h_dim, epoch=epoch, monte=monte)

    # データの準備
    data = load_data()
    data_mats = data_to_mat(data)
    data_mats = torch.tensor(data_mats, dtype=torch.float32).view(-1, 3*NS)
    
    # クロスバリデーションの準備
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_index, test_index in kf.split(data_mats):
        X_train, X_test = data_mats[train_index], data_mats[test_index]
        
        # モデルの学習
        model.train_model(X_train)
        
        # スコアの計算
        score = model.evaluate(X_test)
        scores.append(score)
    
    # 平均スコアを最大化
    scores = np.array(scores)
    return scores.mean()

try:
    study = optuna.load_study(study_name='vae_cv_study', storage='sqlite:///study.db')
except:
    study = optuna.create_study(
        study_name = "vae_cv_study",
        direction='minimize', 
        storage='sqlite:///study.db'
    )
study.optimize(objective, n_trials=100)
print("Best params:", study.best_params)
print("Best value:", study.best_value)

with open("best_params.pkl", "wb") as f:
    pk.dump(study.best_params, f)