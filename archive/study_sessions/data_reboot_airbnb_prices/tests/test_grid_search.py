from nbresult import ChallengeResultTestCase


class TestGridSearch(ChallengeResultTestCase):

    def test_best_score(self):
        self.assertAlmostEqual(self.result.best_score, 0.58, places=2,
                               msg="Your best score of the grid search seems off.")

    def test_best_params(self):
        self.assertIsInstance(self.result.best_params, dict,
                               msg="Your best params should be a dictionary.")
        self.assertEqual(len(self.result.best_params.keys()), 2,
                               msg="Your best params should have 2 keys.")

    def test_test_score(self):
        self.assertAlmostEqual(self.result.test_score, 0.46, places=2,
                               msg="Your score on the test set seems off.")

    def test_training_count(self):
        self.assertGreater(self.result.training_count, 10,
                         msg="Your count is way too low.")
        self.assertGreater(self.result.training_count, 15,
                         msg="Your count is way too low. Think about the CV in GreadSearchCV.")
        self.assertGreater(self.result.training_count, 45,
                         msg="Your count is too low.")
        self.assertLess(self.result.training_count, 80,
                         msg="Your count is too high.")
        self.assertNotEqual(self.result.training_count, 75,
                         msg="Your count is a bit too low. Remember, GridSearchCV also trains a final model on the whole training data.")
        self.assertEqual(self.result.training_count, 76,
                         msg="Your count is wrong.")
