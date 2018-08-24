
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# # Load Stuff

# In[2]:


folderpath = './data/'
filename_test = 'testing'
filename_train = 'training'
filename_black = 'banned_skus'
filename_submission = 'submission'


# In[3]:


df_test = pd.read_csv(folderpath + filename_test + '.csv')
print("TEST Set Loaded")


# In[4]:


df_train = pd.read_csv(folderpath + filename_train + '.csv')
print("Train Set Loaded")

# In[5]:


blacklist = pd.read_csv(folderpath + filename_black + '.csv')


# In[ ]:


# In[10]:


# Subsample of only 1M
df_train_mini = df_train.iloc[:1000000,:]
#df_test_mini = df_test.iloc[:10000,:]


# In[7]:


# How many segments & Shapes
print (np.unique(df_train.segment))
print (df_train.shape)
print (df_train_mini.shape)
print (df_test.shape)





# # Magic 2: Most Popular Skus per Segment

# In[8]:


segments = [1,2,3,4,5,6,7]
SEGMENT_SKUS = []
for seg in segments:
    df_seg = df_train_mini.loc[df_train_mini.segment==seg,:]
    pop_skus = df_seg.loc[:,['sku', 'segment']].groupby('sku').count()
    # delete forbidden items:
    #print (len(pop_skus))
    pop_skus = pop_skus[~pop_skus.index.isin(blacklist.sku)]
    #print (len(pop_skus))
    # Get 20 most popular skus
    selected_skus = pop_skus.sort_values(by=['segment']) 
    selected_skus = selected_skus.iloc[:-21:-1]
    selected_skus = selected_skus.index.tolist()
    SEGMENT_SKUS.append(selected_skus)
print(np.shape(SEGMENT_SKUS))


# # Play around with creating new data-frames
# 
# I hope this would speed up the filtering process of forbidden and duplicate items in the test-set

# In[14]:


# Make Submission File Pandas and filter later...
# Convert data-frame to numpy array so I can filter forbidden testpairs easier
# Make Submission Pandas File
df_submission = pd.DataFrame(columns=['customer_no','sku'])
for seg in segments:
    customer_ids = np.unique(df_test[df_test.segment==seg].customer_no)
    for customer in customer_ids:
        for item in SEGMENT_SKUS[seg-1]:
            df_submission = df_submission.append({'customer_no':customer, 'sku':item}, ignore_index=True)
    print ('segment {} finished'.format(seg))


# In[15]:


testarray = df_test[['customer_no', 'sku']]
print (len(df_submission))
df_submission = df_submission[~df_submission.isin(testarray)].dropna()
print (len(df_submission))


# In[17]:


df_submission.to_csv(folderpath + 'magic2_legal.csv', index=False)


