from nbresult import ChallengeResultTestCase


class TestScaler(ChallengeResultTestCase):

    def test_columns_numeric(self):
        self.assertNotIn('neighbourhood_cleansed', self.result.num_cols,
                         msg="The neighbourhood_cleansed column is a categorical feature.")
        self.assertNotIn('instant_bookable', self.result.num_cols,
                         msg="The instant_bookable column is a categorical feature.")
        self.assertEqual(len(self.result.num_cols), 10,
                         msg="It seems you're picked the wrong number of columns. There should be 10.")


    def test_has_the_right_shape(self):
        self.assertEqual(self.result.X_train_scaled.shape, (2212, 10),
                             msg="You have lost some rows or columns in the train set.")
