# -*- coding: utf-8 -*-
"""
DataSets
=================================

DataSet from `A high-resolution human contact network for infectious disease transmission`.

Source: http://www.pnas.org/content/107/51/22020

"""
#   Rion Brattig Correia <rionbr@gmail.com>
#   All rights reserved.
#   MIT license.
import os
import inspect
import pandas as pd


class base_dataset(object):
    """ Base Dataset class. Handles attributes and methods used for the Salanthé dataset."""

    BASE_URL = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))

    def get_contact_sequence(self):
        """Returns a DataFrame with the Contact Sequence

        Returns: pd.DataFrame
        """
        return self.dfC

    def get_metadata(self):
        """Returns a DataFrame with the dataset Metadata, if available.

        Returns: pd.DataFrame
        """
        return self.dfM

    def __str__(self):
        return "<Dataset(name='%s', size='%d')>" % (self.name, len(self.dfC))


class high_school(base_dataset):
    """ Salanthé American High School Dataset """
    name = 'Salathé High School'

    def __init__(self):
        columns = ['i', 'j', 'count']
        self.dfC = pd.read_csv(self.BASE_URL + '/high-school/sd02.txt', sep='\t', names=columns, encoding='utf-8')

        # Metadata
        self.dfM = pd.read_csv(self.BASE_URL + '/high-school/sd03.txt', sep='\t', index_col=0, names=['i', 'role'])

        # Drop 2 nodes that had no connectivity
        self.dfM.drop(0, inplace=True)
        self.dfM.drop(548, inplace=True)
