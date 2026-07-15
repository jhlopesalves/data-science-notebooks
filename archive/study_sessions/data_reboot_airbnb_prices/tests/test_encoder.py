from nbresult import ChallengeResultTestCase


class TestEncoder(ChallengeResultTestCase):

    def test_columns_categorical(self):
        self.assertEqual(len(self.result.cat_cols), 4,
                         msg="It seems you have the wrong number of categorical columns. There should be 4.")

    def test_has_the_right_shape(self):
        self.assertLess(self.result.X_train_encoded.shape[1], 30,
                            msg="You have way too many columns. Did you set a limit on the number of categories?")
        self.assertNotEqual(self.result.X_train_encoded.shape[1], 24,
                            msg="You have a redundant column. Are you handling binary categories correctly?")
        self.assertEqual(self.result.X_train_encoded.shape, (2212, 23),
                             msg="You have lost some rows or columns in the train set.")
