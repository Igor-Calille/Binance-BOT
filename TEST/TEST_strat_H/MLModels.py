from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np


class using_RandomForestRegressor():
    def __init__(self):
        pass

    def normal_split_RandomForestRegressor(X,y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model

    def GridSearchCV_RandomForestRegressor(X,y):
        tscv = TimeSeriesSplit(n_splits=5)
        model = RandomForestRegressor(random_state=42)

        
        param_search = {
            'n_estimators': [50, 100],
            'max_features': ['sqrt', 'log2'],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }

        gsearch = GridSearchCV(estimator=model, cv=tscv, param_grid=param_search, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
        gsearch.fit(X, y)

        print(f"Best parameters: {gsearch.best_params_}")

        best_model = gsearch.best_estimator_
        
        return best_model
    
    def fixed_params_RandomForestRegressor(X, y):
        # Parâmetros fixos para o modelo
        model = RandomForestRegressor(
            n_estimators=50,
            max_features='sqrt',
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            bootstrap=False,
            random_state=42
        )
        
        # Configuração do TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []

        # Validação cruzada manual usando TimeSeriesSplit
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = mean_squared_error(y_test, y_pred)
            scores.append(score)
        
        # Exibe o MSE médio para as divisões temporais
        print("Média do MSE com TimeSeriesSplit:", np.mean(scores))
        
        # Treine o modelo final nos dados completos
        model.fit(X, y)
        
        return model
    
    