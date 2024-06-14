"""
This script cleans the kdd12 dataset merges the two files rec_log_train.txt
and user_profile.txt and makes 256 batches of 100_000 - 200_000 rows each. 
The batches are saved as pickle files and are to be used in the
Lightgbm_analysis_mpi.py script. The batches are made by randomly sampling
without replacement from merged dataset.
"""

import numpy as np
import pickle
import pandas as pd


rec_log = pd.read_csv('../Data/rec_log_train.txt', sep='\t', header=None)
user_profile = pd.read_csv('../Data/user_profile.txt', sep='\t', header=None)

print(rec_log.shape)
print(rec_log.head())

print(user_profile.shape)
print(user_profile.head())


# find the max number of tags and make tha many columns in the user_profile dataframe
# split the tags into the columns padding with np.nan where there are no tags
max_semi = user_profile[4].str.count(';').max()
tags = user_profile[4].str.split(';', expand=True)

tags = tags.astype(str)
tags.replace('None', np.nan, inplace=True)

# insert tags into user_profile and drop the last column of user_profile
user_profile = user_profile.drop(columns=[4])
user_profile = pd.concat([user_profile, tags], axis=1)

### remove all years with '-' these are unknown years
user_profile.replace(regex=r'[^-]*-[^-]*', value=np.nan, inplace=True)

user_profile = user_profile.astype(float)

def make_batch(seed, N1 = 100_000, N2 = 200_000):
    """
    Function that makes a batch of the kdd12 dataset by randomly sampling
    without replacement from the merged dataset. The seed it important as it
    makes reproducability possible but we can gurentee that the batches are
    different for different seeds. The batches are saved as pickle files.
    The batches are of size N where N is a uniformly distributed random number 
    between N1 and N2.

    Parameters
    ----------
    seed : int
        The seed used for the random number generator.

    N1 : int
        The lower limit of the number of rows in the batch.

    N2 : int
        The upper limit of the number of rows in the batch.

    Returns
    -------
    tot_data : dataframe
        The batch of the kdd12 dataset.
    """

    np.random.seed(seed)

    N = np.random.randint(N1, N2)

    tot_data = pd.DataFrame(np.zeros((N, 26)))

    # take N rows from the rec_log set randomly without replacement
    temp_data = rec_log.sample(n=N, replace=False, random_state=seed)

    tot_data.iloc[:N, 0:3] = temp_data.iloc[:N, 0:3]

    # run through the ids in the rec_log set and match them with the ids in the user profile
    # if they match add the row from the user profile to the tot_data
    # if there is no match print the id
    for i in range(N):
        if i % 10_000 == 0:
            print(f'{i} / {N} done', end ='\r')
        if rec_log.iloc[i, 0] in user_profile.iloc[:, 0].values:
            tot_data.iloc[i, 3:] = user_profile.iloc[np.where(user_profile.iloc[:, 0] == rec_log.iloc[i, 0])[0][0], 1:]
        else:
            print(f'no match for id {rec_log.iloc[i, 0]}')
            tot_data.iloc[i, 3:] = np.zeros(19)
    return tot_data

# make 256 batches 
for i in range(2,256):
    make_batch(i).to_pickle(f'../Data/Batches/kdd12_batch_{i}.pkl')
    print(f'batch {i} / 256 done \n')