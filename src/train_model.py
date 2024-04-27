from IrisModel import IrisModel

# iniciar objeto do tipo IrisModel
iris_model = IrisModel()

# obter os dados de treino 
X_train, X_test, y_train, y_test = iris_model.load_data(test_size=0.2, random_state=42)

# treina o modelo passando o conjunto de treinamento
iris_model.train(X_train, y_train)

# testa o modelo passando o conjunto de teste
result = iris_model.evaluate(X_test, y_test)
print(result)

iris_model.save_model(path='models/iris_model.pkl')