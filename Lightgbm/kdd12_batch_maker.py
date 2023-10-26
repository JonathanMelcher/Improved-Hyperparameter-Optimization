import numpy as np
import pickle
import pandas as pd


big_dataset = pd.read_csv('../Data/rec_log_train.txt', sep='\t', header=None)
user_profile = pd.read_csv('../Data/user_profile.txt', sep='\t', header=None)

print(big_dataset.shape)
print(big_dataset.head())

print(user_profile.shape)
print(user_profile.head())

# find the lagest number of ';' in the last column of the user profile file
max_semi = user_profile[4].str.count(';').max()
# make user profiles last column onto a matrix with N rows and max_semi columns
tags = user_profile[4].str.split(';', expand=True)

# make all entries in the dataframe None into strings 'None' and then replace them with np.nan

# convert None to 'None
tags = tags.astype(str)
tags.replace('None', np.nan, inplace=True)

# insert tags into user_profile and drop the last column of user_profile
user_profile = user_profile.drop(columns=[4])
user_profile = pd.concat([user_profile, tags], axis=1)

### remove all years with '-'

user_profile.replace(regex=r'[^-]*-[^-]*', value=np.nan, inplace=True)

user_profile = user_profile.astype(float)

def make_batch(seed, N1 = 100_000, N2 = 200_000):
    # set the seed to 1999
    np.random.seed(seed)
    # make N a random number between 100_000 and 200_000

    N = np.random.randint(N1, N2)

    # make an empty dataframe N rows and 26 columns
    tot_data = pd.DataFrame(np.zeros((N, 26)))

    # take N rows from the big data set randomly without replacement
    temp_data = big_dataset.sample(n=N, replace=False, random_state=seed)

    tot_data.iloc[:N, 0:3] = temp_data.iloc[:N, 0:3]
    # run through the ids in the big data set and match them with the ids in the user profile
    # if they match add the row from the user profile to the tot_data
    # if there is no match printh the id

    for i in range(N):
        if i % 10_000 == 0:
            print(f'{i} / {N} done', end ='\r')
        if big_dataset.iloc[i, 0] in user_profile.iloc[:, 0].values:
            tot_data.iloc[i, 3:] = user_profile.iloc[np.where(user_profile.iloc[:, 0] == big_dataset.iloc[i, 0])[0][0], 1:]
        else:
            print(f'no match for id {big_dataset.iloc[i, 0]}')
            tot_data.iloc[i, 3:] = np.zeros(19)
    return tot_data

# make 256 batches 
for i in range(2,256):
    make_batch(i).to_pickle(f'../Data/Batches//kdd12_batch_{i}.pkl')
    print(f'batch {i} / 256 done \n')