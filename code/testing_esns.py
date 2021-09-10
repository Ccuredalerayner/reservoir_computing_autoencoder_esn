import random
import time
import datasets
import double_esn
import standard_esn
import Mega_ESN_clean
import torch as torch
import pandas as pd
import csv
import matplotlib.pyplot as plt

def test_main(number_datasets):
    '''
    All seven networks are trained on the same random dataset generated
    by datasets.multi_dataset. Once trained the networks are tested on
    a randomly generated test dataset using he same multi_dataset function.
    This test step is repeated number_datasets times. The outputs of
    each network are appended to a csv file defined below the function.
    :param number_datasets: int number of test sets to be generated
    :return:
    '''
    test_main_start = time.time()
    # train
    # gen data params
    gift_data_train = datasets.multi_dataset(1000)

    print('training...')
    # mega - A B A
    loss_megaABA, file_name_megaABA, esn_loss_a, auto_loss_a, esn_loss_b, auto_loss_b, output_a = Mega_ESN_clean.Mega_esn(
        feedback=True,
        load=None,
        train=True,
        gift_data=gift_data_train,
        washout=[0],
        hidden_size=30,
        input_size_a=1,
        input_size_b=2,
        output_size_a=2,
        output_size_b=3,
        loss_fcn=torch.nn.MSELoss())
    row = [file_name_megaABA, '', 'mega_ABA', number_datasets, 'yes', 'train', loss_megaABA]
    print(row)
    writer.writerow(row)

    # mega - A B
    '''
    loss_megaAB, file_name_megaAB, esn_loss_a, auto_loss_a, esn_loss_b, auto_loss_b, output_a = Mega_ESN_clean.Mega_esn(
        feedback=False,
        load=None,
        train=True,
        gift_data=gift_data_train,
        washout=[0],
        hidden_size=30,
        input_size_a=1,
        input_size_b=2,
        output_size_a=2,
        output_size_b=3,
        loss_fcn=torch.nn.MSELoss())
    row = [file_name_megaAB, '', 'mega_AB', number_datasets, 'no', 'train', loss_megaAB]
    writer.writerow(row)'''

    # standard - 30 no feed
    loss, file_name_standard_30_no_feed = standard_esn.single_esn_testing(washout=[0],
                                                                          hidden_size=30,
                                                                          input_size=1,
                                                                          output_size=3,
                                                                          load=None,
                                                                          loss_f=torch.nn.MSELoss(),
                                                                          feedback=False,
                                                                          train=True,
                                                                          gift_data=gift_data_train)

    row = [file_name_standard_30_no_feed, '', 'standard_30', number_datasets, 'no', 'train', loss]
    print(row)
    writer.writerow(row)
    # standard - 30 feed
    loss, file_name_standard_30_feed = standard_esn.single_esn_testing(washout=[0],
                                                                       hidden_size=30,
                                                                       input_size=1,
                                                                       output_size=3,
                                                                       load=None,
                                                                       loss_f=torch.nn.MSELoss(),
                                                                       feedback=True,
                                                                       train=True,
                                                                       gift_data=gift_data_train)

    row = [file_name_standard_30_feed, '', 'standard_30', number_datasets, 'yes', 'train', loss]
    print(row)
    writer.writerow(row)
    # standard - 60 no feed
    loss, file_name_standard_60_no_feed = standard_esn.single_esn_testing(washout=[0],
                                                                          hidden_size=60,
                                                                          input_size=1,
                                                                          output_size=3,
                                                                          load=None,
                                                                          loss_f=torch.nn.MSELoss(),
                                                                          feedback=False,
                                                                          train=True,
                                                                          gift_data=gift_data_train)

    row = [file_name_standard_60_no_feed, '', 'standard_60', number_datasets, 'no', 'train', loss]
    print(row)
    writer.writerow(row)
    # standard - 60 feed
    loss, file_name_standard_60_feed = standard_esn.single_esn_testing(washout=[0],
                                                                       hidden_size=60,
                                                                       input_size=1,
                                                                       output_size=3,
                                                                       load=None,
                                                                       loss_f=torch.nn.MSELoss(),
                                                                       feedback=True,
                                                                       train=True,
                                                                       gift_data=gift_data_train)

    row = [file_name_standard_60_feed, '', 'standard_60', number_datasets, 'yes', 'train', loss]
    print(row)
    writer.writerow(row)
    # double - 30 no feed
    loss, file_name_double_A_30_no_feed, file_name_double_B_30_no_feed = double_esn.double_esn_testing(feedback=False,
                                                                                                       load=None,
                                                                                                       load_b=None,
                                                                                                       train=True,
                                                                                                       gift_data=gift_data_train,
                                                                                                       washout=[0],
                                                                                                       hidden_size=30,
                                                                                                       input_size_a=1,
                                                                                                       input_size_b=2,
                                                                                                       output_size_a=2,
                                                                                                       output_size_b=3,
                                                                                                       loss_fcn=torch.nn.MSELoss())
    row = [file_name_double_A_30_no_feed, file_name_double_B_30_no_feed, 'double_30', number_datasets, 'no', 'train', loss]
    print(row)
    writer.writerow(row)
    # double - 30 feed
    loss, file_name_double_A_30_feed, file_name_double_B_30_feed = double_esn.double_esn_testing(feedback=True,
                                                                                                 load=None,
                                                                                                 load_b=None,
                                                                                                 train=True,
                                                                                                 gift_data=gift_data_train,
                                                                                                 washout=[0],
                                                                                                 hidden_size=30,
                                                                                                 input_size_a=1,
                                                                                                 input_size_b=2,
                                                                                                 output_size_a=2,
                                                                                                 output_size_b=3,
                                                                                                 loss_fcn=torch.nn.MSELoss())
    row = [file_name_double_A_30_feed, file_name_double_B_30_feed, 'double_30', number_datasets, 'yes', 'train', loss]
    print(row)
    writer.writerow(row)

    #################################################################################################
    # test
    print('testing...')
    for p in range(number_datasets):
        gift_data_test = datasets.multi_dataset(1000)
        # mega - A B A
        loss_megaABA, file_name_megaABA, esn_loss_a, auto_loss_a, esn_loss_b, auto_loss_b, output_a = Mega_ESN_clean.Mega_esn(
        feedback=True,
        load=file_name_megaABA,
        train=False,
        gift_data=gift_data_test,
        washout=[0],
        hidden_size=30,
        input_size_a=1,
        input_size_b=2,
        output_size_a=2,
        output_size_b=3,
        loss_fcn=torch.nn.MSELoss())
        row = [file_name_megaABA, '', 'mega_ABA', number_datasets, 'yes', 'test', loss_megaABA]
        writer.writerow(row)
        print(row)

        # mega - A B
        '''
        loss_megaAB, file_name_megaAB, esn_loss_a, auto_loss_a, esn_loss_b, auto_loss_b, output_a = Mega_ESN_clean.Mega_esn(
        feedback=False,
        load=None,
        train=False,
        gift_data=gift_data_test,
        washout=[0],
        hidden_size=30,
        input_size_a=1,
        input_size_b=2,
        output_size_a=2,
        output_size_b=3,
        loss_fcn=torch.nn.MSELoss())
        row = [file_name_megaAB, '', 'mega_AB', number_datasets, 'no', 'test', loss_megaAB]
        writer.writerow(row)'''

    # standard - 30 no feed
        loss, file_name_standard_30_no_feed = standard_esn.single_esn_testing(washout=[0],
                                                                          hidden_size=30,
                                                                          input_size=1,
                                                                          output_size=3,
                                                                          load=file_name_standard_30_no_feed,
                                                                          loss_f=torch.nn.MSELoss(),
                                                                          feedback=False,
                                                                          train=False,
                                                                          gift_data=gift_data_test)

        row = [file_name_standard_30_no_feed, '', 'standard_30', number_datasets, 'no', 'test', loss]
        print(row)
        writer.writerow(row)
    # standard - 30 feed
        loss, file_name_standard_30_feed = standard_esn.single_esn_testing(washout=[0],
                                                                       hidden_size=30,
                                                                       input_size=1,
                                                                       output_size=3,
                                                                       load=file_name_standard_30_feed,
                                                                       loss_f=torch.nn.MSELoss(),
                                                                       feedback=True,
                                                                       train=False,
                                                                       gift_data=gift_data_test)

        row = [file_name_standard_30_feed, '', 'standard_30', number_datasets, 'yes', 'test', loss]
        print(row)
        writer.writerow(row)
        # standard - 60 no feed
        loss, file_name_standard_60_no_feed = standard_esn.single_esn_testing(washout=[0],
                                                                          hidden_size=60,
                                                                          input_size=1,
                                                                          output_size=3,
                                                                          load=file_name_standard_60_no_feed,
                                                                          loss_f=torch.nn.MSELoss(),
                                                                          feedback=False,
                                                                          train=False,
                                                                          gift_data=gift_data_test)

        row = [file_name_standard_60_no_feed, '', 'standard_60', number_datasets, 'no', 'test', loss]
        print(row)
        writer.writerow(row)
        # standard - 60 feed
        loss, file_name_standard_60_feed = standard_esn.single_esn_testing(washout=[0],
                                                                       hidden_size=60,
                                                                       input_size=1,
                                                                       output_size=3,
                                                                       load=file_name_standard_60_feed,
                                                                       loss_f=torch.nn.MSELoss(),
                                                                       feedback=True,
                                                                       train=False,
                                                                       gift_data=gift_data_test)

        row = [file_name_standard_60_feed, '', 'standard_60', number_datasets, 'yes', 'test', loss]
        print(row)
        writer.writerow(row)
        # double - 30 no feed
        loss, file_name_double_A_30_no_feed, file_name_double_B_30_no_feed = double_esn.double_esn_testing(feedback=False,
                                                                                                           load=file_name_double_A_30_no_feed,
                                                                                                           load_b=file_name_double_B_30_no_feed,
                                                                                                           train=False,
                                                                                                           gift_data=gift_data_test,
                                                                                                           washout=[0],
                                                                                                           hidden_size=30,
                                                                                                           input_size_a=1,
                                                                                                           input_size_b=2,
                                                                                                           output_size_a=2,
                                                                                                           output_size_b=3,
                                                                                                           loss_fcn=torch.nn.MSELoss())
        row = [file_name_double_A_30_no_feed, file_name_double_B_30_no_feed, 'double_30', number_datasets, 'no', 'test', loss]
        print(row)
        writer.writerow(row)
        # double - 30 feed
        loss, file_name_double_A_30_feed, file_name_double_B_30_feed = double_esn.double_esn_testing(feedback=True,
                                                                                                     load=file_name_double_A_30_feed,
                                                                                                     load_b=file_name_double_B_30_feed,
                                                                                                     train=False,
                                                                                                     gift_data=gift_data_test,
                                                                                                     washout=[0],
                                                                                                     hidden_size=30,
                                                                                                     input_size_a=1,
                                                                                                     input_size_b=2,
                                                                                                     output_size_a=2,
                                                                                                     output_size_b=3,
                                                                                                     loss_fcn=torch.nn.MSELoss())
        row = [file_name_double_A_30_feed, file_name_double_B_30_feed, 'double_30', number_datasets, 'yes', 'test', loss]
        print(row)
        writer.writerow(row)
        print("test_main ended in", time.time() - test_main_start, "seconds.")

# run main
# open the file in the write mode
file = 'saves/test_3/trial_test_3.csv'
f = open(file, 'w')
# create the csv writer
writer = csv.writer(f)
col = ['file name A', 'file name B', 'esn type', 'number datasets', 'feedback', 'test/train', 'loss']
# write a row to the csv file
writer.writerow(col)

start = time.time()
for j in range(50):
    print(j)
    test_main(50)

print("program ended in", time.time() - start, "seconds.")

# close the file
f.close()

save = pd.read_csv(file, sep=',')
print(save)
