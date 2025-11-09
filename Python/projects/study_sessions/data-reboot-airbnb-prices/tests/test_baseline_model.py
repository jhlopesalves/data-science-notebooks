from nbresult import ChallengeResultTestCase


class TestBaselineModel(ChallengeResultTestCase):

    def test_is_better_than_baseline(self):
        self.assertIsNotNone(self.result.better_than_baseline,
                             msg="Store your answer yes or no as a string in better_than_baseline")
        self.assertIsInstance(self.result.better_than_baseline, str,
                              msg="Store your answer yes or no as a string in better_than_baseline")
        self.assertIn(self.result.better_than_baseline.lower(), ["yes", "no"],
                        msg="Choose one of the values yes or no.")
        self.assertEqual(self.result.better_than_baseline.lower(), "yes",
                            msg="Your model is better than the baseline. Remember that a higher R2 is better.")
