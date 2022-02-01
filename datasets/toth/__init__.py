# -*- coding: utf-8 -*-
"""
DataSets
=================================

DataSet from `The role of heterogeneity in contact timing and duration in network models of influenza spread in schools`.

Source: http://rsif.royalsocietypublishing.org/content/12/108/20150279

"""
#   Rion Brattig Correia <rionbr@gmail.com>
#   All rights reserved.
#   MIT license.
import inspect
import os
import pandas as pd
import datetime


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


class elementary_school(base_dataset):
    """ Salanthé High School Dataset

    Args:
        date (string): Either date '2013-01-31' (day 1) or '2013-02-01 (day 2) of measured contacts.
        distance (int): Either None (no distance qualification) or 1 (one-meter contact data)

    Note:
        The value of 19.53125 comes from the paper methods.
    """
    name = 'Toth Elementary School'

    def __init__(self, date, distance=None):
        columns = ['i', 'j', 'created_time', 'length']

        if date == 'all':
            dates = ['2013-01-31', '2013-02-01']
        else:
            dates = [date]

        if distance is None:
            if date == '2013-01-31':
                contact_file_names = ['D3 - Elem1 day 1 contact data.txt']
            elif date == '2013-02-01':
                contact_file_names = ['D4 - Elem1 day 2 contact data.txt']
            elif date == 'all':
                contact_file_names = ['D3 - Elem1 day 1 contact data.txt', 'D4 - Elem1 day 2 contact data.txt']
            else:
                raise ValueError("Date must be either '2013-01-31' or '2013-02-01'")
        elif distance == 1:
            if date == '2013-01-31':
                contact_file_names = ['D9 - Elem1 day 1 one-meter contact data.txt']
            elif date == '2013-02-01':
                contact_file_names = ['D10 - Elem1 day 2 one-meter contact data.txt']
            elif date == 'all':
                contact_file_names = ['D9 - Elem1 day 1 one-meter contact data.txt', 'D10 - Elem1 day 2 one-meter contact data.txt']
            else:
                raise ValueError("Date must be either '2013-01-31' or '2013-02-01'")
        else:
            raise ValueError("Distance must be either None or 1 (one-meter)")

        dfCs = []
        for date, contact_file_name in zip(dates, contact_file_names):
            dfC = pd.read_csv(self.BASE_URL + '/elementary-school/' + contact_file_name, sep=' ', header=0, names=columns, encoding='utf-8')
            # Records have a length of contact, we need to expand them to a record for each 20S contact
            records = dfC.to_records(index=False)
            expanded = [(i, j, created_time + x) for i, j, created_time, length in records for x in range(length)]
            dfC = pd.DataFrame.from_records(expanded, columns=['i', 'j', 'created_time'])

            # the timedelta between contacts
            timedelta = pd.to_timedelta(dfC['created_time'] * 19.53125, unit='s')
            dfC['created_time_fmt'] = pd.to_datetime(datetime.datetime.strptime(date, '%Y-%m-%d') + timedelta).dt.round('S')
            dfCs.append(dfC)
        self.dfC = pd.concat(dfCs, axis=0)

        # Metadata
        self.dfM = pd.read_csv(self.BASE_URL + '/elementary-school/D6 - Elem1 student data.txt', sep=' ', header=0, index_col=0, names=['i', 'grade', 'class', 'gender'])
        # Update metadata
        self.dfM['gender'].replace(to_replace={0: 'Male', 1: 'Female'}, inplace=True)
        self.dfM['grade'].replace(to_replace={0: 'Kindergarden', -1: 'Unknown'}, inplace=True)


class middle_school(base_dataset):
    """ Salanthé Middle School Dataset

    Args:
        date (string): Either date '2012-11-28' (day 1) or '2012-11-29 (day 2) of measured contacts.
        distance (int): Either None (no distance qualification) or 1 (one-meter contact data)

    Note:
        The value of 19.53125 comes from the paper methods.
    """
    name = 'Toth Elementary School'

    def __init__(self, date, distance=None):
        columns = ['i', 'j', 'created_time', 'length']

        if date == 'all':
            dates = ['2012-11-28', '2012-11-29']
        else:
            dates = [date]

        if distance is None:
            if date == '2012-11-28':
                contact_file_names = ['D1 - Mid1 day 1 contact data.txt']
            elif date == '2012-11-29':
                contact_file_names = ['D2 - Mid1 day 2 contact data.txt']
            elif date == 'all':
                contact_file_names = ['D2 - Mid1 day 2 contact data.txt', 'D1 - Mid1 day 1 contact data.txt']
            else:
                raise ValueError("Date must be either '2012-11-28' or '2012-11-29'")
        elif distance == 1:
            if date == '2012-11-28':
                contact_file_names = ['D7 - Mid1 day 1 one-meter contact data.txt']
            elif date == '2012-11-29':
                contact_file_names = ['D8 - Mid1 day 2 one-meter contact data.txt']
            elif date == 'all':
                contact_file_names = ['D7 - Mid1 day 1 one-meter contact data.txt', 'D8 - Mid1 day 2 one-meter contact data.txt']
            else:
                raise ValueError("Date must be either '2012-11-28' or '2012-11-29'")
        else:
            raise ValueError("Distance must be either None or 1 (one-meter)")

        dfCs = []
        for date, contact_file_name in zip(dates, contact_file_names):

            dfC = pd.read_csv(self.BASE_URL + '/middle-school/' + contact_file_name, sep=' ', header=0, names=columns, encoding='utf-8')
            # Records have a length of contact, we need to expand them to a record for each 20S contact
            records = dfC.to_records(index=False)
            expanded = [(i, j, created_time + x) for i, j, created_time, length in records for x in range(length)]

            dfC = pd.DataFrame.from_records(expanded, columns=['i', 'j', 'created_time'])

            # the timedelta between contacts
            timedelta = pd.to_timedelta(dfC['created_time'] * 19.53125, unit='s')
            dfC['created_time_fmt'] = pd.to_datetime(datetime.datetime.strptime(date, '%Y-%m-%d') + timedelta).dt.round('S')
            dfCs.append(dfC)
        self.dfC = pd.concat(dfCs, axis=0)

        # Metadata
        self.dfM = pd.read_csv(self.BASE_URL + '/middle-school/D5 - Mid1 student data.txt', sep=' ', header=0, index_col=0, names=['i', 'grade', 'gender'])
        # Update metadata
        self.dfM['gender'].replace(to_replace={0: 'Male', 1: 'Female', -1: 'Unknown'}, inplace=True)
        self.dfM['grade'].replace(to_replace={-1: 'Unknown'}, inplace=True)
