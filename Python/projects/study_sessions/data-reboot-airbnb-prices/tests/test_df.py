from nbresult import ChallengeResultTestCase


class TestDf(ChallengeResultTestCase):
    def test_df_has_no_double_index_column(self):
        self.assertNotIn("Unnamed: 0", self.result.df_columns,
                         msg="df has a redundant index column")

    def test_price_has_the_right_dtype(self):
        self.assertEqual(self.result.price_dtype, 'float64',
                         msg="The price column has the wrong dtype")
