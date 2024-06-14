# Improved-Hyperparameter-Optimization-
Code repository for the article "Improved Hyperparameter Optimization and Multi-Dimensional Sampling"


### LGBM analysis

The LGBM analysis is made by first using kdd12_batch_maker.py to generate
batches of cleaned combined data. These have different sizes. From here the 
Lightgbm_analysis_mpi.py file is run, preferably on a large cluster, using the
pipings to control what dimensions and resolutions to run and how many workers
to use. The results from this analysis are then finally compressed down to the 
data needed for the plots with Results_Compression.py. Here the results coming from some grids have nonuniform
size and taking the mean and standard deviation is therefore not possible with 
array operations. 


### Data for LGBM analysis
The data is available here https://www.kaggle.com/competitions/kddcup2012-track1/data.
The files needed are rec_log_train.txt and user_profile.txt 

The batches made from the KDD12 dataset with varying sizes and cleaned data for the LGBM analysis are found here https://sid.erda.dk/cgi-sid/ls.py?share_id=hqbY6ZtoDE .

The raw and compressed results are hosted here https://sid.erda.dk/cgi-sid/ls.py?share_id=d2NdsN3a7p .