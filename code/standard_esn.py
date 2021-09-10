import gc

import matplotlib.pyplot as plt
import now as now
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchesn.nn import ESN
from torchesn import utils
import time
from tqdm import tqdm
from datetime import datetime
import datasets


def single_esn_testing(data_size=1000, washout=[0], hidden_size=60, input_size=1, output_size=3, load=None,
                       loss_f=torch.nn.MSELoss(), feedback=False, train=False, gift_data=None,
                       c=0.1, d=0.1, e=0.1, c_change=0.001, d_change=0.001, e_change=0.001):
    '''
    :param data_size: int data points in dataset
    :param washout: [int] randomly removes some datapoints
    :param hidden_size: int hidden size of the network
    :param input_size: int input size for network
    :param output_size: int output size for network
    :param load: String the file name of the ESN network
    :param loss_f: pytorch loss function for calculation loss and training
    :param feedback: bool True, feedback loop on network False, no feedback
    :param train: Bool True, the network will be trained. False, the network will not be trained
    :param gift_data: list [0] = y data, [1] = a,b data, [2] = c,d,e data
    :param c: float dataset parameter
    :param d: float dataset parameter
    :param e: float dataset parameter
    :param c_change: float dataset parameter
    :param d_change: float dataset parameter
    :param e_change: float dataset parameter
    :return: loss: float loss for network
    :return: file_name: string file name for network save
    '''
    device = torch.device('cuda')
    dtype = torch.double
    torch.set_default_dtype(dtype)

    loss_fcn = loss_f

    if feedback:
        input_size = input_size + hidden_size

    # load file
    if load is not None:
        file_name = load  # file_name = "saves/" + load

        model_esn = ESN(input_size, hidden_size, output_size,
                        output_steps='mean', readout_training='cholesky').to(device)
        model_esn.load_state_dict(torch.load(file_name))  # it takes the loaded dictionary, not the path file itself
        model_esn.eval()
    else:
        model_esn = ESN(input_size, hidden_size, output_size).to(device)

    # Data
    if gift_data is not None:
        X_data, Y_data, Z_data = gift_data[0],gift_data[1],gift_data[2]
        data_size = len(X_data)
    else:
        X_data, Y_data, Z_data = datasets.sin_link(data_size, c, d, e, c_change, d_change, e_change)
    X_data = torch.tensor(X_data).to(device)
    Y_data = torch.tensor(Y_data).to(device)
    Z_data = torch.tensor(Z_data).to(device)
    X_data = X_data.reshape(-1, 1, 1)
    Y_data = Y_data.reshape(-1, 1, 2)
    Z_data = Z_data.reshape(-1, 1, 3)

    # for hidden loop
    if feedback:
        X_data = F.pad(input=X_data, pad=(0, hidden_size, 0, 0), mode='constant', value=0)

    guess_flat_b = utils.prepare_target(Z_data.clone(), [X_data.size(0)], washout)

    if (load is None) or train:
        # for hidden layer pass
        hidden_layer = torch.zeros(1, 1, input_size).to(device)
        for data, answer in tqdm(zip(X_data, guess_flat_b)):
            data = data.reshape(1, 1, -1)
            # for hidden layer pass
            if feedback:
                data = torch.add(data, hidden_layer)
            answer = answer.reshape(1, -1)

            model_esn(data, washout, None, answer)  # check if None changes hidden output
            model_esn.fit()

            # for hidden layer pass
            if feedback:
                _, hidden = model_esn(data, washout)
                hidden_layer = F.pad(input=hidden, pad=(1, 0, 0, 0), mode='constant', value=0)

    # error
    output, hidden = model_esn(X_data, washout)
    loss = loss_fcn(output, Z_data[washout[0]:]).item()

    if load is None:
        if feedback:
            file_loop = 'C_loop'
        else:
            file_loop = 'C'
        date = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        file_name = "saves/ESN_" + file_loop + '_' + date + ".pth"
        torch.save(model_esn.state_dict(), file_name)

    return loss, file_name

# print(f'feedback is false: {single_esn_testing(feedback=True,load="ESN_C_loop_31-08-2021_19-37-29.pth")}')
# print(f'feedback is true: {single_esn_testing(feedback=True, )}')
