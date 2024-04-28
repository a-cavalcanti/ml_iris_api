from unittest import TestCase, main
import numpy as np
from sklearn.datasets import load_iris
from src.IrisModel import IrisModel, IrisFeatures

class TestIrisModel(TestCase):
    
    def setUp(self):
        self.iris_model = IrisModel()
        self.X_train, self.X_test, self.y_train, self.y_test = self.iris_model.load_data()
        self.iris_model.train(self.X_train, self.y_train)
        self.results = self.iris_model.evaluate(self.X_test, self.y_test)
        self.features = IrisFeatures(sepal_length=5.9, sepal_width=3.0, petal_length=5.1, petal_width=1.8)

    def test_load_data(self):
        self.assertEqual(self.X_train.shape[0], 120)  # 80% of 150
        self.assertEqual(self.X_test.shape[0], 30)   # 20% of 150
        self.assertEqual(len(self.y_train), 120)
        self.assertEqual(len(self.y_test), 30)

    def test_train(self):
        self.assertIsNotNone(self.iris_model.model)

    def test_evaluate(self):
        results = self.iris_model.evaluate(self.X_test, self.y_test)
        self.assertIsInstance(results, dict)
        self.assertTrue('accuracy' in results)
        self.assertTrue('precision' in results)
        self.assertTrue('recall' in results)
        self.assertTrue('f1_score' in results)

    def test_predict(self):
        prediction = self.iris_model.predict(self.features)
        self.assertIsInstance(prediction, int)

    def test_save_and_load_model(self):
        self.iris_model.save_model('tests/test_iris_model.pkl')
        self.iris_model.model = None  # Reset model to ensure loading works
        self.iris_model.load_model('tests/test_iris_model.pkl')
        self.assertIsNotNone(self.iris_model.model)

        # Test prediction after loading to ensure model was saved and loaded correctly
        prediction = self.iris_model.predict(self.features)
        self.assertIsInstance(prediction, int)

    def test_accuracy(self):
        self.assertGreaterEqual(self.results['accuracy'], 0.8)

    def test_precision(self):
        self.assertGreaterEqual(self.results['precision'], 0.8)

    def test_recall(self):
        self.assertGreaterEqual(self.results['recall'], 0.8)

    def test_f1_score(self):
        self.assertGreaterEqual(self.results['f1_score'], 0.8)

    def test_prediction(self):
        prediction = self.iris_model.predict(self.features)
        self.assertIsInstance(prediction, int)

    def test_class_name_map(self):        
        self.assertEqual(self.iris_model.class_name_map(0), 'setosa')
        self.assertEqual(self.iris_model.class_name_map(1), 'versicolor')
        self.assertEqual(self.iris_model.class_name_map(2), 'virginica')

if __name__ == '__main__':
    main()
