import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
'''
After opening the desired csv file, a formula is grated to 
specifically select each network. This is later used in 
all_data_list to plot a box plot of all points using 
matplotlib. The second plot instead takes all of the test 
data and plots it for each of the test sets. And the final 
plot is the sum of each networks losses for a given test set.
'''

# saves/test_3/trial_test_2.csv
# file name A,file name B,esn type,number datasets,feedback,test/train,loss

df = pd.read_csv('saves/test_3/trial_test_2.csv')

pd.set_option('display.max_rows', None)

# mega
filter_mega_test = (df['esn type'] == 'mega_ABA') & (df['test/train'] == 'test')

filter_mega_test_data = pd.DataFrame(df.loc[filter_mega_test, 'loss'])
filter_mega_test_list = np.hstack(filter_mega_test_data.values)

# standard_30
filter_standard_30_feedback_test = (df['esn type'] == 'standard_30') & (df['feedback'] == 'yes') & (df['test/train'] == 'test')
filter_standard_30_no_feedback_test = (df['esn type'] == 'standard_30') & (df['feedback'] == 'no') & (df['test/train'] == 'test')

filter_standard_30_no_feedback_test_data = pd.DataFrame(df.loc[filter_standard_30_no_feedback_test, 'loss'])
filter_standard_30_no_feedback_test_list = np.hstack(filter_standard_30_no_feedback_test_data.values)

filter_standard_30_feedback_test_data = pd.DataFrame(df.loc[filter_standard_30_feedback_test, 'loss'])
filter_standard_30_feedback_test_list = np.hstack(filter_standard_30_feedback_test_data.values)

# standard_60
filter_standard_60_feedback_test = (df['esn type'] == 'standard_60') & (df['feedback'] == 'yes') & (df['test/train'] == 'test')
filter_standard_60_no_feedback_test = (df['esn type'] == 'standard_60') & (df['feedback'] == 'no') & (df['test/train'] == 'test')

filter_standard_60_no_feedback_test_data = pd.DataFrame(df.loc[filter_standard_60_no_feedback_test, 'loss'])
filter_standard_60_no_feedback_test_list = np.hstack(filter_standard_60_no_feedback_test_data.values)

filter_standard_60_feedback_test_data = pd.DataFrame(df.loc[filter_standard_60_feedback_test, 'loss'])
filter_standard_60_feedback_test_list = np.hstack(filter_standard_60_feedback_test_data.values)

# double_30
filter_double_30_feedback_test = (df['esn type'] == 'double_30') & (df['feedback'] == 'yes') & (df['test/train'] == 'test')
filter_double_30_no_feedback_test = (df['esn type'] == 'double_30') & (df['feedback'] == 'no') & (df['test/train'] == 'test')

filter_double_30_no_feedback_test_data = pd.DataFrame(df.loc[filter_double_30_no_feedback_test, 'loss'])
filter_double_30_no_feedback_test_list = np.hstack(filter_double_30_no_feedback_test_data.values)

filter_double_30_feedback_test_data = pd.DataFrame(df.loc[filter_double_30_feedback_test, 'loss'])
filter_double_30_feedback_test_list = np.hstack(filter_double_30_feedback_test_data.values)


all_data_list = np.dstack((filter_mega_test_list,
               filter_standard_30_no_feedback_test_list,
               filter_standard_30_feedback_test_list,
               filter_standard_60_no_feedback_test_list,
               filter_standard_60_feedback_test_list,
               filter_double_30_no_feedback_test_list,
               filter_double_30_feedback_test_list,
               ))

all_data_list = all_data_list.reshape(-1,7)

plt.rcParams["figure.figsize"] = (18, 8)
fig1, ax1 = plt.subplots()
ax1.set_title('Box plot showing each ESN type and the data collected from 50 iterations of 50 tests')
ax1.boxplot(pd.DataFrame(all_data_list), vert=False, notch=True)
ax1.set_xlabel('Mean Squared Error')
ax1.set_yticklabels(['mega','standard 30 no feedback','standard 30 feedback','standard 60 no feedback',
                     'standard 60 feedback','double 30 no feedback','double 30 feedback'])
plt.show()

filter_datasets_test = ((df['test/train'] == 'test'))
filter_datasets_data = pd.DataFrame(df.loc[filter_datasets_test, 'loss'])
filter_datasets_list = np.hstack(filter_datasets_data.values)
filter_datasets_list_reshape_dataset_wise = filter_datasets_list.reshape(-1, 50)

plt.rcParams["figure.figsize"] = (8, 15)
fig2, ax2 = plt.subplots()
ax2.set_title('Box plot showing each test set of data against loss')
ax2.boxplot(pd.DataFrame(filter_datasets_list_reshape_dataset_wise), vert=False, notch=True)
ax2.set_xlabel('Mean Squared Error')
ax2.set_ylabel('Test set')
plt.show()

filter_datasets_list_reshape_network_wise = filter_datasets_list.reshape(7,50,50)
a = np.zeros([7, 50])
for network, i in zip(filter_datasets_list_reshape_network_wise, range(7)):
    for data, j in zip(network, range(50)):
        a[i][j] = sum(data)


plt.rcParams["figure.figsize"] = (18, 8)
fig3, ax3 = plt.subplots()
ax3.set_title('Box plot showing total loss for each network')
ax3.boxplot(pd.DataFrame(a.reshape(50,7)), vert=False, notch=True)
ax3.set_xlabel('Total Mean Squared Error for Each network')
ax3.set_yticklabels(['mega','standard 30 no feedback','standard 30 feedback','standard 60 no feedback',
                     'standard 60 feedback','double 30 no feedback','double 30 feedback'])
plt.show()
