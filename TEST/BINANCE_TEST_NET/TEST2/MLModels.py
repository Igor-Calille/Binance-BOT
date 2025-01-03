import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error

class using_RandomForestRegressor:
    def __init__(self):
        pass

    @staticmethod
    def normal_split_RandomForestRegressor(X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model

    @staticmethod
    def GridSearchCV_RandomForestRegressor(X, y):
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

        gsearch = GridSearchCV(
            estimator=model, cv=tscv, param_grid=param_search,
            scoring='neg_mean_squared_error', n_jobs=-1, verbose=2
        )
        gsearch.fit(X, y)
        print(f"Best parameters: {gsearch.best_params_}")
        best_model = gsearch.best_estimator_
        return best_model

    @staticmethod
    def fixed_params_RandomForestRegressor(X, y):
        """
        Modelo Random Forest com parâmetros fixos.
        Faz uma validação cruzada com TimeSeriesSplit
        e depois treina no dataset completo.
        """
        model = RandomForestRegressor(
            n_estimators=50,
            max_features='sqrt',
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            bootstrap=False,
            random_state=42
        )

        tscv = TimeSeriesSplit(n_splits=5)
        scores = []

        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = mean_squared_error(y_test, y_pred)
            scores.append(score)

        # Treina no dataset inteiro
        model.fit(X, y)
        return model

    @staticmethod
    def partial_fit_approach(X, y, model=None, batch_size=50):
        """
        Exemplo de função 'partial_fit' para fins didáticos.
        Não está sendo usada no app.py.
        """
        # Se model for None, crie um novo
        if model is None:
            model = RandomForestRegressor(
                n_estimators=10,
                random_state=42,
                warm_start=True
            )

        # Realiza 'chunks' de treino
        n = len(X)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            X_batch = X.iloc[start:end]
            y_batch = y.iloc[start:end]

            # warm_start do RandomForest no sklearn não é igual a partial_fit,
            # mas aqui é apenas ilustrativo. Modelos como SGDRegressor suportam partial_fit de verdade.
            if start == 0:
                model.fit(X_batch, y_batch)
            else:
                model.n_estimators += 1  # aumenta numero de árvores a cada batch
                model.fit(X_batch, y_batch)

        return model
