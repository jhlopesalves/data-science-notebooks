from nbresult import ChallengeResultTestCase


class TestReduced(ChallengeResultTestCase):
    def test_reduced_min_max(self):
        self.assertEqual(self.result.reduced_min, 50,
                         msg="You should include prices greater than or EQUAL TO 50.")
        self.assertEqual(self.result.reduced_max, 1500,
                         msg="You should include prices less than or EQUAL TO 1500.")

    def test_reduced_has_the_right_shape(self):
        self.assertEqual(self.result.reduced_shape, (2766, 21))
