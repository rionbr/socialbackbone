# -*- coding: utf-8 -*-
"""
SocioPatterns.org DataSets
=================================

Datasets from the SocioPatterns.org project.

Source: http://www.sociopatterns.org/datasets/
"""
#   Rion Brattig Correia <rionbr@gmail.com>
#   Nathan David Ratkiewicz <nratkiew@indiana.edu>
#   All rights reserved.
#   MIT license.
import inspect
import os
import numpy as np
import pandas as pd
import datetime
try:
    import cStringIO as StringIO
except Exception as e:
    import StringIO as StringIO
import tarfile


class base_dataset(object):
    """ Base Dataset class. Handles attributes and methods used for all SocioPatterns datasets."""

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


class exhibit(base_dataset):
    """ SocioPatterns Exhibit dataset
    Exhibit experiment ran from `2009-04-28` to `2009-07-17`

    Args:
        date (string or list of strings, optional): return only a certain date or date range. String format is 'YYYY-MM-DD'.
    """
    name = 'SocioPatterns Exhibit'

    def __init__(self, date=None):
        columns = ['created_time', 'i', 'j']

        # Return a specific day, a date range, or everything?
        if isinstance(date, str):
            dates = [ datetime.datetime.strptime(date, '%Y-%m-%d').date()]
        elif isinstance(date, list):
            dates = [ datetime.datetime.strptime(d).date() for d in date]
        else:
            dates = list()

        # Loading gz files from tar
        content = str()
        tar = tarfile.open(self.BASE_URL + '/exhibit/sg_infectious_contact_list.tgz','r:gz')
        for member in tar.getmembers():
            file_name = member.name
            file_date = datetime.datetime.strptime(file_name[13:23], '%Y_%m_%d').date()

            if (file_date in dates) or (date is None):
                f = tar.extractfile(member)
                if f is not None:
                    content += f.read()
        self.dfC = pd.read_csv(StringIO.StringIO(content), sep='\t', names=columns)
        # Created_time is already the datetime in seconds (utc)
        self.dfC['created_time_fmt'] = pd.to_datetime(self.dfC['created_time'], unit='s')
        # Metadata
        users = np.unique(self.dfC[['i', 'j']].values)
        self.dfM = pd.DataFrame({'label': users}, index=users)


class high_school(base_dataset):
    """ SocioPatterns High School Dataset """
    name = 'SocioPatterns High School'

    def __init__(self):
        columns = ['created_time', 'i', 'j', 'type_i', 'type_j']
        self.dfC = pd.read_csv(self.BASE_URL + '/high-school/High-School_data_2013.csv.gz', sep=' ', names=columns, encoding='utf-8')

        # Created_time is already the datetime in seconds (utc)
        self.dfC['created_time_fmt'] = pd.to_datetime(self.dfC['created_time'], unit='s')
        # Metadata
        self.dfM = pd.read_csv(self.BASE_URL + '/high-school/metadata_2013.txt', sep='\t', index_col=0, names=['i', 'class', 'gender'])
        self.dfM['label'] = self.dfM.index
        # In the high_school dataset, metadata users [478,2] had no contact sequence.
        # Therefore they are dropped so that the network only has one connected component.
        self.dfM.drop(478, inplace=True)
        self.dfM.drop(2, inplace=True)


class hospital(base_dataset):
    """ SocioPatterns Hospital Dataset """
    name = 'SocioPatterns Hospital'

    def __init__(self):
        columns = ['created_time', 'i', 'j', 'type_i', 'type_j']
        year, month, day, hour, minute, second = 2010, 12, 06, 13, 00, 00
        self.dfC = pd.read_csv(self.BASE_URL + '/hospital/detailed_list_of_contacts_Hospital.dat.gz', sep='\t', names=columns, encoding='utf-8')
        # the timedelta between contacts
        timedelta = pd.to_timedelta(self.dfC['created_time'], unit='s')
        self.dfC['created_time_fmt'] = pd.to_datetime(datetime.datetime(year, month, day, hour, minute, second) + timedelta)
        # Metadata
        self.dfM = pd.concat([self.dfC[['i', 'type_i']], self.dfC[['j', 'type_j']].rename(columns={'j': 'i', 'type_j': 'type_i'})], axis=0, ignore_index=True)
        self.dfM = self.dfM.drop_duplicates(subset=['i', 'type_i'], keep='first').set_index('i').rename(columns={'type_i': 'type'})
        self.dfM['label'] = self.dfM.index


class conference(base_dataset):
    """ SocioPatterns Hypertext 2009 Conference Dataset """
    name = 'SocioPatterns Hypertext 2009 Conference'

    def __init__(self):
        columns = ['created_time', 'i', 'j']
        year, month, day = 2009, 06, 29
        #
        self.dfC = pd.read_csv(self.BASE_URL + '/conference/ht09_contact_list.dat.gz', sep='\t', names=columns, encoding='utf-8')
        # the timedelta between contacts
        timedelta = pd.to_timedelta(self.dfC['created_time'], unit='s')
        self.dfC['created_time_fmt'] = pd.to_datetime(datetime.datetime(year, month, day) + timedelta)
        # Metadata
        users = np.unique(self.dfC[['i', 'j']].values)
        self.dfM = pd.DataFrame({'label': users}, index=users)


class primary_school(base_dataset):
    """ SocioPatterns Primary School Dataset """
    name = 'SocioPatterns Primary School'

    def __init__(self):
        columns = ['created_time', 'i', 'j', 'type_i', 'type_j']
        year, month, day = 2009, 10, 01
        #
        self.dfC = pd.read_csv(self.BASE_URL + '/primary-school/primaryschool.csv.gz', sep='\t', names=columns, encoding='utf-8')
        # the timedelta between contacts
        timedelta = pd.to_timedelta(self.dfC['created_time'], unit='s')
        self.dfC['created_time_fmt'] = pd.to_datetime(datetime.datetime(year, month, day) + timedelta)
        # Metadata
        self.dfM = pd.read_csv(self.BASE_URL + '/primary-school/metadata_primaryschool.txt', sep='\t', index_col=0, names=['i', 'class', 'gender'])
        self.dfM['label'] = self.dfM.index


class workplace(base_dataset):
    """ SocioPatterns Workplace Dataset

    Args:
        size (string): The size of the network to return. Either 'short' (13) or 'large' (15).
    """
    name = 'SocioPatterns Workplace'

    def __init__(self, size='large'):
        columns = ['created_time', 'i', 'j']
        year, month, day = 2013, 06, 24
        #
        if size == 'short':
            datafile = self.BASE_URL + '/workplace/tij_InVS13.dat.gz'
            metafile = self.BASE_URL + '/workplace/metadata_InVS13.dat'
        elif size == 'large':
            datafile = self.BASE_URL + '/workplace/tij_InVS15.dat.gz'
            metafile = self.BASE_URL + '/workplace/metadata_InVS15.dat'
        else:
            raise ValueError("The value of attribute 'size' must be either 'short' or 'large'.")
        self.dfC = pd.read_csv(datafile, sep='\t', names=columns, encoding='utf-8')

        # the timedelta between contacts
        timedelta = pd.to_timedelta(self.dfC['created_time'], unit='s')
        self.dfC['created_time_fmt'] = pd.to_datetime(datetime.datetime(year, month, day) + timedelta)
        # Metadata
        self.dfM = pd.read_csv(metafile, sep='\t', index_col=0, names=['i', 'class'])
        self.dfM['label'] = self.dfM.index


if __name__ == '__main__':

    d = primary_school()
    dfC = d.get_contact_sequence()
    print dfC.head()

    dfM = d.get_metadata()
    print dfM.head()
