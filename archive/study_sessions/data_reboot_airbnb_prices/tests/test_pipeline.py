from nbresult import ChallengeResultTestCase


class TestPipeline(ChallengeResultTestCase):

    def test_has_the_right_shape(self):
        '''Test that the pipeline does what it should do
        '''
        self.assertEqual(self.result.X_train_to_test_pipeline.shape, (2212, 33),
                         msg="Your pipeline does not give the expected results.")

    def test_train_was_transformed(self):
        '''Tests that the preprocessor was actually applied to X_train
        and saved in X_train_preprocessed
        '''
        self.assertEqual(self.result.X_train_preprocessed.shape,
                         self.result.X_train_to_test_pipeline.shape,
                         msg="Did you fit and transform your preprocessor pipeline to X_train?")

    def test_test_was_only_transformed(self):
        '''Tests that the preprocessor was not fitted on the test set
        '''
        self.assertEqual(self.result.X_test_preprocessed.shape,
                         self.result.X_test_to_test_pipeline_not_fitted_on_test.shape,
                         msg="Did you fit your preprocessor pipeline to X_test? Think again.")
