import numpy as np
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from pydantic import BaseModel


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class IrisModel:
    """
    Uma classe para encapsular as operações do modelo de classificação Iris, 
    incluindo treinamento, avaliação e previsão.
    """

    def __init__(self):
        """
        Inicializa o IrisModel com um modelo vazio e um StandardScaler 
        para escalonamento de dados.
        """
        self.model = None
        self.scaler = StandardScaler()

    def load_data(self, test_size=0.2, random_state=42):
        """
        Carrega o conjunto de dados Iris e o divide em conjuntos de treinamento e teste.

        Returns:
            tuple: Uma tupla contendo conjuntos de treinamento e teste (X_train, X_test, 
            y_train, y_test).
        """
        data = load_iris()
        return train_test_split(data.data, data.target, test_size=test_size, random_state=random_state)

    def train(self, X_train, y_train):
        """
        Treina o modelo de regressão logística usando os dados de treinamento 
        dimensionados.

        Parameters:
            X_train (array): Features dos dados de treinamento.
            y_train (array): Classes de dados de treinamento.
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model = LogisticRegression(max_iter=200)
        self.model.fit(X_train_scaled, y_train)

    def evaluate(self, X_test, y_test):
        """
        Avalia o modelo usando os dados de teste e calcula as principais métricas de 
        classificação.

        Parameters:
            X_test (array): Features dos dados de teste.
            y_test (array): Classes dos dados de teste.

        Returns:
            dict: Um dicionário com acurácia, precisão, recall e f1_score.
        """
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='macro'
        )
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    def predict(self, features: IrisFeatures):
        """
        Prevê a classe do Iris com base nas features de entrada.

        Parameters:
            features (IrisFeatures): As features de entrada agrupadas no modelo pydantic IrisFeatures.

        Returns:
            int: Classe predita.
        """
        input_data = np.array([
            [features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]
        ])
        input_scaled = self.scaler.transform(input_data)
        return int(self.model.predict(input_scaled)[0])

    def save_model(self, path='iris_model.pkl'):
        """
        Salva o modelo treinado em um arquivo.

        Parameters:
            path (str): Caminho do arquivo onde o modelo será salvo.
        """
        with open(path, 'wb') as file:
            pickle.dump(self.model, file)

    def load_model(self, path='iris_model.pkl'):
        """
        Lê o modelo de um arquivo.

        Parameters:
            path (str): Caminho do arquivo de onde o modelo será carregado.
        """
        with open(path, 'rb') as file:
            self.model = pickle.load(file)
