from nbresult import ChallengeResultTestCase


class TestLinregPipe(ChallengeResultTestCase):

    def test_cv_score(self):
        self.assertIsInstance(self.result.cv_score, float,
                               msg="Your score is in the wrong format. Did you take the average of the cv scores?")
        self.assertAlmostEqual(self.result.cv_score, 0.51, places=2,
                               msg="Your score seems very much off.")
