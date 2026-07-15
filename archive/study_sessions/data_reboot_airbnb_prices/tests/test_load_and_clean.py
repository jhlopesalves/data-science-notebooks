from nbresult import ChallengeResultTestCase


class TestLoadAndClean(ChallengeResultTestCase):
    def test_return(self):
        self.assertIsNotNone(self.result.new_df,
                             msg="Does your function return something")

    def test_cleaned_has_the_right_shape(self):
        self.assertEqual(self.result.new_shape, (2766, 21))

    def test_price_has_the_right_dtype(self):
        self.assertEqual(self.result.price_dtype, 'float64',
                         msg="The price column has the wrong dtype")

    def test_bathrooms_text_dtype(self):
          self.assertEqual(self.result.bathrooms_text_dtype, 'float64',
                         msg="The bathrooms_text column has the wrong dtype")

    def test_instant_bookable_dtype(self):
          self.assertEqual(self.result.instant_bookable_dtype, 'int64',
                         msg="The instant_bookable column has the wrong dtype")
