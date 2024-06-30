import unittest
import pandas as pd
import numpy as np
import split_merge as sm

class TestPrepareSequencesDF(unittest.TestCase):
    def setUp(self):
        self.df_in = pd.DataFrame({
            'open_date':['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05'],
            'code':['A', 'A', 'A', 'A', 'A'],
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [6, 7, 8, 9, 10],
            'feature3': [11, 12, 13, 14, 15],
            'label': [0, 1, 0, 1, 0]
        })
        self.df_in['open_date'] = pd.to_datetime(self.df_in['open_date'])
        self.df_in = self.df_in.set_index(['open_date','code'])
        self.list_features = ['feature1', 'feature2', 'feature3']
        self.sequence_length = 3
        self.str_new_col = 'sequence'

    def test_prepare_sequences_df(self):
        df_out = sm.prepare_sequences_df(df_in=self.df_in,list_features= self.list_features, sequence_length=self.sequence_length, str_new_col=self.str_new_col)
        print(df_out)
        # Check that the output DataFrame has the correct columns
        self.assertListEqual(list(df_out.columns), self.list_features +['label'] + [self.str_new_col])

        # Check that the sequences contain the correct values
        correct_values =[None,
                        None,
                        [[1, 6, 11], [2, 7, 12], [3,8,13]],
                        [[2, 7, 12], [3, 8, 13], [4,9,14]],
                        [[3, 8, 13], [4, 9, 14], [5,10,15]]]
        
        # dont be confused by the index, it is just a placeholder
        index_to_int = {index: i for i, index in enumerate(df_out.index)}

        for i, row in df_out.iterrows():
            for j in range(self.sequence_length):
                # None is a placeholder for the first two rows
                if correct_values[index_to_int[i]] is None:
                    self.assertIsNone(row[self.str_new_col])
                else:
                    self.assertListEqual(list(row[self.str_new_col][j]), correct_values[index_to_int[i]][j])

if __name__ == '__main__':
    unittest.main()