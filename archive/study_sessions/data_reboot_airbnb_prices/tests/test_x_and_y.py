from nbresult import ChallengeResultTestCase


class TestXAndY(ChallengeResultTestCase):

    def test_x_has_the_right_shape(self):
        self.assertEqual(self.result.x_shape, (2766, 14),
                             msg="You have too many columns left")

    def test_y_has_the_right_shape(self):
        self.assertEqual(self.result.y_shape, (2766, ),
                             msg="Your Y has the wrong shape. It should be a series.")

    def test_target_not_in_x(self):
        self.assertNotIn('price', self.result.x_columns,
                         msg="You should remove the target from your X.")

    def test_excluded_columns(self):
        for col in ['id', 'listing_url', 'scrape_id', 'last_scraped', 'name',
                    'description']:
            self.assertNotIn(col, self.result.x_columns,
                             msg=f"The {col} column is not useful.")
