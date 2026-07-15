from nbresult import ChallengeResultTestCase


class TestCleaned(ChallengeResultTestCase):
    def test_cleaned_has_the_right_shape(self):
        self.assertEqual(self.result.reduced_shape, (2766, 21))

    def test_bathrooms_text_dtype(self):
          self.assertEqual(self.result.bathrooms_text_dtype, 'float64',
                         msg="The bathrooms_text column has the wrong dtype")

    def test_instant_bookable_dtype(self):
          self.assertEqual(self.result.instant_bookable_dtype, 'int64',
                         msg="The instant_bookable column has the wrong dtype")
