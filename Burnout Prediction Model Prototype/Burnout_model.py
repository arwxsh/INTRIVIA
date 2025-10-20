import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')
os.makedirs("models", exist_ok=True)


class BurnoutModel:
    def __init__(self):
        self.psych_models = {}
        self.burnout_models = {} 
        self.scaler = None


        self.easy_vars = [
            'sex','age','year','part','job','health','stud_h',
            'age2','age_job_inter','year_part_inter','health_job_inter'
        ]


        self.psych_vars = [
            'stress_index','empathy_balance','motivation_score',
            'stai_t','cesd','psyt','qcae_aff','qcae_cog',
            'amsp','jspe','erec_mean'
        ]
        
        self.burnout_vars = ['mbi_ex', 'mbi_cy', 'mbi_ea']


    def preprocess_data(self, path):
        data = pd.read_csv(path)


        data['stress_index'] = (0.6 * data['stai_t'] + 0.8 * data['cesd']) * (1 + 0.15 * data['psyt']) ** 1.1
        data['empathy_balance'] = (data['qcae_aff'] - 0.9 * data['qcae_cog']) * (1 + 0.1 * data['sex']) / (1 + 0.03 * data['year'])
        data['motivation_score'] = (data['amsp'] / (1 + 0.5 * (data['stai_t'] / 50) ** 2)) * (1 + 0.07 * data['stud_h']) / (1 + 0.015 * data['age'])


        all_vars = ['sex','age','year','part','job','health','stud_h'] + self.psych_vars + self.burnout_vars
        data = data[all_vars].dropna()


        data['age2'] = data['age'] ** 2
        data['age_job_inter'] = data['age'] * data['job']
        data['year_part_inter'] = data['year'] * data['part']
        data['health_job_inter'] = data['health'] * data['job']


        return data


    def train(self, path):
        data = self.preprocess_data(path)


        X_easy = data[self.easy_vars]
        Y_psych = data[self.psych_vars]
        Y_burnout = data[self.burnout_vars]


        X_train, X_test, Y_psych_train, Y_psych_test, Y_burnout_train, Y_burnout_test = train_test_split(
            X_easy, Y_psych, Y_burnout, test_size=0.2, random_state=42
        )


        psych_preds = pd.DataFrame(index=Y_psych_test.index)
        psych_r2_scores = []


        cat_feature_indices = [0, 2, 3, 4, 5]
        
        for col in self.psych_vars:
            model = CatBoostRegressor(
                iterations=300, depth=6, learning_rate=0.05,
                loss_function='RMSE', verbose=0, random_seed=42
            )
            model.fit(X_train, Y_psych_train[col], cat_features=cat_feature_indices)
            preds = model.predict(X_test)
            psych_preds[col] = preds
            psych_r2_scores.append(r2_score(Y_psych_test[col], preds))
            self.psych_models[col] = model


        print(f"Stage 1: Easy → Psychological (avg R²): {np.mean(psych_r2_scores):.3f}")

        self.scaler = StandardScaler()
        Y_psych_scaled = self.scaler.fit_transform(Y_psych)


        X_train_p, X_test_p, y_train_b, y_test_b = train_test_split(
            Y_psych_scaled, Y_burnout, test_size=0.2, random_state=42
        )


        burnout_r2_scores = {}
        for burnout_var in self.burnout_vars:
            model = CatBoostRegressor(
                iterations=400, depth=6, learning_rate=0.05,
                loss_function='RMSE', verbose=0, random_seed=42
            )
            model.fit(X_train_p, y_train_b[burnout_var])
            preds = model.predict(X_test_p)
            r2 = r2_score(y_test_b[burnout_var], preds)
            burnout_r2_scores[burnout_var] = r2
            self.burnout_models[burnout_var] = model
            print(f"Stage 2: Psychological → {burnout_var.upper()} (R²): {r2:.3f}")

        psych_full_preds = pd.DataFrame(index=data.index)
        for col in self.psych_vars:
            psych_full_preds[col] = self.psych_models[col].predict(X_easy)


        psych_full_scaled = self.scaler.transform(psych_full_preds)
        
        print("\nStage 3: Final Chain (Easy → Burnout) Approx. R²:")
        for burnout_var in self.burnout_vars:
            burnout_pred = self.burnout_models[burnout_var].predict(psych_full_scaled)
            final_r2 = r2_score(Y_burnout[burnout_var], burnout_pred)
            print(f"  {burnout_var.upper()}: {final_r2:.3f}")


        self.save()
        print("\nTraining complete — models saved to 'models/' folder.")


    def save(self):
        for name, model in self.psych_models.items():
            model.save_model(f"models/psych_model_{name}.cbm")
        for name, model in self.burnout_models.items():
            model.save_model(f"models/burnout_model_{name}.cbm")
        joblib.dump(self.scaler, "models/psych_scaler.pkl")


    def load(self):
        from catboost import CatBoostRegressor
        self.psych_models = {}
        for col in self.psych_vars:
            m = CatBoostRegressor()
            m.load_model(f"models/psych_model_{col}.cbm")
            self.psych_models[col] = m
        
        self.burnout_models = {}
        for burnout_var in self.burnout_vars:
            m = CatBoostRegressor()
            m.load_model(f"models/burnout_model_{burnout_var}.cbm")
            self.burnout_models[burnout_var] = m
            
        self.scaler = joblib.load("models/psych_scaler.pkl")


    def predict_burnout(self, X_easy_new):
        X_easy_new = X_easy_new.copy()


        cat_cols = ['sex', 'year', 'part', 'job', 'health']
        numeric_cols = ['age', 'stud_h']
        
        cat_numeric = X_easy_new[cat_cols].copy()
        for col in cat_cols:
            cat_numeric[col] = pd.to_numeric(cat_numeric[col], errors='coerce')
        
        for col in cat_cols:
            X_easy_new[col] = X_easy_new[col].astype(str)
        
        for col in numeric_cols:
            X_easy_new[col] = X_easy_new[col].astype(float)


        X_easy_new['age2'] = X_easy_new['age'] ** 2
        X_easy_new['age_job_inter'] = X_easy_new['age'] * cat_numeric['job']
        X_easy_new['year_part_inter'] = cat_numeric['year'] * cat_numeric['part']
        X_easy_new['health_job_inter'] = cat_numeric['health'] * cat_numeric['job']


        X_easy_new = X_easy_new[self.easy_vars]


        psych_preds = pd.DataFrame({
            col: model.predict(X_easy_new) for col, model in self.psych_models.items()
        })



        psych_scaled = self.scaler.transform(psych_preds)


        burnout_predictions = {}
        for burnout_var in self.burnout_vars:
            burnout_predictions[burnout_var] = self.burnout_models[burnout_var].predict(psych_scaled)


        return burnout_predictions
