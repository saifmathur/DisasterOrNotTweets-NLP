# %%
import numpy as np
import pandas as pd
from sklearn import feature_extraction,linear_model,model_selection,preprocessing

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

#%%
train_df[train_df["target"] == 0]["text"].values[1]
train_df[train_df["target"] == 1]["text"].values[1]

cv = feature_extraction.text.CountVectorizer()
#getting counts for the first 5 tweets in the data
example_train_vectors = cv.fit_transform(train_df["text"][0:5]) #first 5 records

"""since these vectors are mostly sparse we`re gonna
    remove the zero elements to save space
"""
#%%
#%%

print(example_train_vectors[0].todense().shape)
print(example_train_vectors[0].todense())

"""
The above code tells us that:
1. There are 54 unique words (or tokens) in the first five tweets
2. The first tweet contains only some of those unique tokens
    -all of the non zero counts above are the tokens that 
    do exist in the first tweet
"""

#%%





#%%