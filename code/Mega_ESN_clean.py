import gc
import now as now
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


# date and time


# Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # N, 784
        self.encoder = nn.Sequential(
            nn.Linear(30, 16),  # N, 784 -> N, 128
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3))

        self.decoder = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 30))

    def forward(self, x):
        encoded = self.encoder(x)
        decoder = self.decoder(encoded)
        return decoder


# train autoencoder
def train_auto(data, model, optimizer, epochs=200, criterion=nn.MSELoss()):
    '''
    :param data: [float] data to train autoencoder on
    :param model: pytorch autoencoder model to train
    :param optimizer: pytorch optimizer optimiser to train autoencoder
    :param epochs: int number of training loops
    :param criterion: pytorch loss function calculates loss of network to train
    :return: loss: float loss of network
    '''
    loss = 0
    for epoch in range(epochs):
        recon = model(data)
        loss = criterion(recon, data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss


def train_esn(train_input, train_output, esn_train, autoencoder_train, autoencoder_optimizer, hidden_pad,
              washout=[0], device='cuda', loss_fcn=nn.MSELoss()):
    '''
    :param train_input: [float] input data to ESN
    :param train_output: [float] desired output from network
    :param esn_train: pytorch-esn ESN network to train
    :param autoencoder_train: pytorch autoencoder model to train
    :param autoencoder_optimizer: pytorch optimizer optimiser to train autoencoder
    :param hidden_pad: int padding for hidden layer feedback (size of input layer)
    :param washout: [int] randomly removes some data points
    :param device: string cuda = if cuda gpu available cpu if no gpu
    :param loss_fcn: pytorch loss function calculates loss of network to train
    :return: esn_loss: float loss of ESN on training set
    :return: auto_loss: float loss of autoencoder on hidden state
    '''
    if washout is None:
        washout = [0]
    guess_flat = utils.prepare_target(train_output.clone(), [train_input.size(0)], washout)
    hidden_layer = torch.zeros(1, 1, train_input.size(2)).to(device)
    auto_loss = []
    for data, answer in tqdm(zip(train_input, guess_flat)):
        data = data.reshape(1, 1, -1)
        data = torch.add(data, hidden_layer)
        answer = answer.reshape(1, -1)

        esn_train(data, washout, None, answer)
        esn_train.fit()

        _, hidden = esn_train(data, washout)
        hidden_layer = F.pad(input=hidden, pad=(hidden_pad, 0, 0, 0), mode='constant', value=0)

        auto_loss = train_auto(hidden, autoencoder_train, autoencoder_optimizer, epochs=150)

    # Training Test
    output, hidden = esn_train(train_input, washout)
    esn_loss = loss_fcn(output, train_output[washout[0]:]).item()
    return esn_loss, auto_loss


def save_network(esn, auto, file_type, date):
    '''
    :param esn: pytorch-esn ESN network to save
    :param auto: pytorch autoencoder model to save
    :param file_type: string denoting the A or B ESN to preserve the order of the network
    :param date: string date and time added to file name to add distinction
    :return:
    '''
    filename_auto = "saves/Autoencoder_" + file_type + '_' + date + ".pth"
    filename_esn = "saves/ESN_" + file_type + '_' + date + ".pth"
    torch.save(esn.state_dict(), filename_esn)
    torch.save(auto.state_dict(), filename_auto)


def load_network(file_name, type_a=True, device='cuda', hidden_size=30, input_size_a=1,
                 input_size_b=2, output_size_a=2, output_size_b=3):
    '''
    :param file_name: string denoting the file to load
    :param type_a: bool True, loading an A network False, loading a B network
    :param device: string cuda = if cuda gpu available cpu if no gpu
    :param hidden_size: int hidden size of the network
    :param input_size_a: int input size for first network
    :param input_size_b: int input size for second network
    :param output_size_a: int output size for first network
    :param output_size_b: int output size for second network
    :return: model_esn_a_loaded: pytorch-esn ESN network loaded
    :return: model_auto_a_loaded: pytorch autoencoder model loaded
    '''
    input_size_a = input_size_a + hidden_size
    input_size_b = input_size_b + hidden_size
    file_auto_load = "saves/Autoencoder_" + file_name + ".pth"
    file_esn_load = "saves/ESN_" + file_name + ".pth"

    model_auto_loaded = Autoencoder().to(device)
    model_auto_loaded.load_state_dict(
        torch.load(file_auto_load))  # it takes the loaded dictionary, not the path file itself
    model_auto_loaded.eval()

    if type_a:
        model_esn_loaded = ESN(input_size_a, hidden_size, output_size_a, output_steps='mean',
                                 readout_training='cholesky').to(device)
    else:
        model_esn_loaded = ESN(input_size_b, hidden_size, output_size_b, output_steps='mean',
                                 readout_training='cholesky').to(device)
    model_esn_loaded.load_state_dict(
        torch.load(file_esn_load))  # it takes the loaded dictionary, not the path file itself
    model_esn_loaded.eval()

    return model_esn_loaded, model_auto_loaded


def generate_data(data_size=1000, c=0.1, d=0.1, e=0.1, c_change=0.001, d_change=0.001, e_change=0.001):
    '''
    :param data_size: int Number of generated dataset values
    :param c: float initial starting value
    :param d: float initial starting value
    :param e: float initial starting value
    :param c_change: float time step change for c param
    :param d_change: float time step change for c param
    :param e_change: float time step change for c param
    :return: X_data: [float] y values for dataset
    :return: Y_data: [[float]] a, b values for dataset
    :return: Z_data: [[float]] c, d, e values for data
    :return: Y_data_b: [[float]] a, b values for dataset manipulated for input of ESN B
    '''
    X_data, Y_data, Z_data = datasets.sin_link(data_size, c, d, e, c_change, d_change, e_change)
    X_data = torch.tensor(X_data).to('cuda')
    Y_data = torch.tensor(Y_data).to('cuda')
    Z_data = torch.tensor(Z_data).to('cuda')
    X_data = X_data.reshape(-1, 1, 1)
    Y_data = Y_data.reshape(-1, 1, 2)
    Z_data = Z_data.reshape(-1, 1, 3)
    X_data = F.pad(input=X_data, pad=(0, 30, 0, 0), mode='constant', value=0)
    Y_data_b = F.pad(input=Y_data, pad=(0, 30, 0, 0), mode='constant', value=0)
    return X_data, Y_data, Z_data, Y_data_b

def pass_data(X_data, Y_data, Z_data):
    '''
    :return: X_data: [float] y values for dataset
    :return: Y_data: [[float]] a, b values for dataset
    :return: Z_data: [[float]] c, d, e values for data
    :return: X_data: [float] y values for dataset
    :return: Y_data: [[float]] a, b values for dataset
    :return: Z_data: [[float]] c, d, e values for data
    :return: Y_data_b: [[float]] a, b values for dataset manipulated for input of ESN B
    '''
    X_data = torch.tensor(X_data).to('cuda')
    Y_data = torch.tensor(Y_data).to('cuda')
    Z_data = torch.tensor(Z_data).to('cuda')
    X_data = X_data.reshape(-1, 1, 1)
    Y_data = Y_data.reshape(-1, 1, 2)
    Z_data = Z_data.reshape(-1, 1, 3)
    X_data = F.pad(input=X_data, pad=(0, 30, 0, 0), mode='constant', value=0)
    Y_data_b = F.pad(input=Y_data, pad=(0, 30, 0, 0), mode='constant', value=0)
    return X_data, Y_data, Z_data, Y_data_b


# trains brand new networks
def train_new_network(X_data, Y_data, Z_data, Y_data_b,
                      model_auto_a, optimizer_a, model_auto_b, optimizer_b,
                      model_esn_a, model_esn_b):
    '''
    :return: X_data: [float] y values for dataset
    :return: Y_data: [[float]] a, b values for dataset
    :return: Z_data: [[float]] c, d, e values for data
    :return: Y_data_b: [[float]] a, b values for dataset manipulated for input of ESN B
    :param model_auto_a: pytorch autoencoder model A
    :param optimizer_a: pytorch optimizer optimiser to train autoencoder A
    :param model_auto_b: pytorch autoencoder model B
    :param optimizer_b: pytorch optimizer optimiser to train autoencoder B
    :param model_esn_a: pytorch-esn ESN network A
    :param model_esn_b: pytorch-esn ESN network B
    :return: model_esn_a: pytorch-esn ESN network A
    :return: model_auto_a: pytorch autoencoder model A
    :return: model_esn_b: pytorch-esn ESN network B
    :return: model_auto_b: pytorch autoencoder model B
    :return: init_date_time: string time/date networks were created
    :return: esn_loss_a: float loss of ESN A on training set
    :return: auto_loss_a: float loss of autoencoder A on hidden state
    :return: esn_loss_b: float loss of ESN B on training set
    :return: auto_loss_b: float loss of autoencoder B on hidden state
    '''
    init_date_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    print(f'Mega training esn and auto: A')
    esn_loss_a, auto_loss_a = train_esn(X_data, Y_data, model_esn_a, model_auto_a, optimizer_a, 1)

    print(f'Mega training esn and auto: B')
    esn_loss_b, auto_loss_b = train_esn(Y_data_b, Z_data, model_esn_b, model_auto_b, optimizer_b, 2)

    save_network(model_esn_a, model_auto_a, 'A', init_date_time)
    save_network(model_esn_b, model_auto_b, 'B', init_date_time)
    return model_esn_a, model_auto_a, model_esn_b, model_auto_b, init_date_time, \
           esn_loss_a, auto_loss_a, esn_loss_b, auto_loss_b


# loading
def load_files(load_date):  # 31-08-2021_16-42-49
    '''
    :param load_date: string file name for loading
    :return: model_esn_a: pytorch-esn ESN loaded network A
    :return: model_auto_a: pytorch autoencoder loaded model A
    :return: model_esn_b: pytorch-esn ESN loaded network B
    :return: model_auto_b: pytorch autoencoder loaded model B
    '''
    # A file
    load_file = 'A_' + load_date
    model_esn_a, model_auto_a = load_network(load_file, type_a=True)

    # B file
    load_file = 'B_' + load_date
    model_esn_b, model_auto_b = load_network(load_file, type_a=False)
    return model_esn_a, model_auto_a, model_esn_b, model_auto_b

# This assumes two esn and respective autoencoders are loaded and ready
def Mega_esn(feedback=False, load=None, train=True, gift_data=None,
             data_size=1000, washout=[0], hidden_size=30, input_size_a=1,
             input_size_b=2, output_size_a=2, output_size_b=3, loss_fcn=torch.nn.MSELoss(),
             c=0.1, d=0.1, e=0.1, c_change=0.001, d_change=0.001, e_change=0.001):
    '''
    :param feedback: Bool True, feedback a normal mega-ESN False, feedback as usual but not loop from B to A
    :param load: String the file name of the ESN pair
    :param train: Bool True, the network will be trained. False, the network will not be trained
    :param gift_data: list [0] = y data, [1] = a,b data, [2] = c,d,e data
    :param data_size: int data points in dataset
    :param washout: [int] randomly removes some datapoints
    :param hidden_size: int hidden size of the network
    :param input_size_a: int input size for first network
    :param input_size_b: int input size for second network
    :param output_size_a: int output size for first network
    :param output_size_b: int output size for second network
    :param loss_fcn: pytorch loss function for calculation loss and training
    :param c: float dataset parameter
    :param d: float dataset parameter
    :param e: float dataset parameter
    :param c_change: float dataset parameter
    :param d_change: float dataset parameter
    :param e_change: float dataset parameter
    :return: loss float total loss of the hole network
    :return: file_data_and_time string file name to reload
    :return: esn_loss_a float loss of esn a
    :return: auto_loss_a float loss of aoutoencoder a
    :return: esn_loss_b float loss of esn b
    :return: auto_loss_b float loss of aoutoencoder b
    :return: output_b_save [[float]] saved output from ESN B
    '''
    # misc
    device = torch.device('cuda')
    dtype = torch.double
    torch.set_default_dtype(dtype)

    # autoncoder setup
    model_auto_a = Autoencoder().to(device)
    model_auto_b = Autoencoder().to(device)
    optimizer_a = torch.optim.Adam(model_auto_a.parameters(), lr=0.001)
    optimizer_b = torch.optim.Adam(model_auto_b.parameters(), lr=0.001)

    # esn setup
    input_size_a = input_size_a + hidden_size
    input_size_b = input_size_b + hidden_size
    model_esn_a = ESN(input_size_a, hidden_size, output_size_a).to(device)
    model_esn_b = ESN(input_size_b, hidden_size, output_size_b).to(device)

    # Main Loop
    start = time.time()

    if gift_data is not None:
        X_data, Y_data, Z_data, Y_data_b = pass_data(gift_data[0],gift_data[1],gift_data[2])
        data_size = len(X_data)
    else:
        X_data, Y_data, Z_data, Y_data_b = generate_data(data_size, c, d, e, c_change, d_change, e_change)

    esn_loss_a, auto_loss_a, esn_loss_b, auto_loss_b = 0, 0, 0, 0
    if load is not None:
        model_esn_a, model_auto_a, model_esn_b, model_auto_b = load_files(load)
        file_data_and_time = load
        if train:
            model_esn_a, model_auto_a, model_esn_b, model_auto_b, file_data_and_time, \
            esn_loss_a, auto_loss_a, esn_loss_b, auto_loss_b = train_new_network(X_data, Y_data,
                                                                                 Z_data, Y_data_b,
                                                                                 model_auto_a,
                                                                                 optimizer_a,
                                                                                 model_auto_b,
                                                                                 optimizer_b,
                                                                                 model_esn_a,
                                                                                 model_esn_b)
    else:
        model_esn_a, model_auto_a, model_esn_b, model_auto_b, file_data_and_time, \
        esn_loss_a, auto_loss_a, esn_loss_b, auto_loss_b = train_new_network(X_data, Y_data,
                                                                             Z_data, Y_data_b,
                                                                             model_auto_a,
                                                                             optimizer_a,
                                                                             model_auto_b,
                                                                             optimizer_b,
                                                                             model_esn_a,
                                                                             model_esn_b)

    data_a_x = X_data
    data_a_y = Y_data

    guess_flat_a = utils.prepare_target(data_a_y.clone(), [data_a_x.size(0)], washout)

    data_b_x = Y_data
    data_b_y = Z_data
    guess_flat_b = utils.prepare_target(data_b_y.clone(), [data_b_x.size(0)], washout)

    a_loss = []
    b_loss = []
    output_b_save = torch.zeros(data_size, 1, 3)
    i = 0

    # construct full loop
    # One by one
    # swap autoencoders encoder and decoders
    # auto_a encoder + auto_b decoder (and vice vera)
    model_auto_ab = Autoencoder()
    model_auto_ba = Autoencoder()

    model_auto_ab.encoder = model_auto_a.encoder
    model_auto_ba.encoder = model_auto_b.encoder

    model_auto_ab.decoder = model_auto_b.decoder
    model_auto_ba.decoder = model_auto_a.decoder

    # esn_a output + autoencoder(a-b) to esn_b,
    # esn_b output + autoencoder(b-a) to esn_a.
    hidden_layer_a = torch.zeros(1, 1, data_a_x.size(2)).to(device)
    for data, answer_a, answer_b in zip(data_a_x, guess_flat_a, guess_flat_b):
        data = data.reshape(1, 1, -1)
        data = torch.add(data, hidden_layer_a)
        answer_a = answer_a.reshape(1, -1)

        # run datapoint on esn_a
        output_a, hidden_a = model_esn_a(data, washout)

        # mid testing
        a_loss.append(loss_fcn(output_a, answer_a[washout[0]:]).item())

        # add padding to hidden layer so we can add input
        hidden_layer_a = F.pad(input=hidden_a, pad=(1, 0, 0, 0), mode='constant', value=0)

        # auto_a(ab)
        # passing esn_a hidden states though new auto_ab
        auto_ab_recon = model_auto_ab(hidden_layer_a)
        # adding esn_a output to autoencoder processed esn_a hidden layer
        output_a = F.pad(input=output_a, pad=(0, 30, 0, 0), mode='constant', value=0)
        auto_ab_recon = F.pad(input=auto_ab_recon, pad=(2, 0, 0, 0), mode='constant', value=0)
        data_b_input = torch.add(output_a, auto_ab_recon)

        # # run datapoint on esn_b
        output_b, hidden_b = model_esn_b(data_b_input, washout)
        output_b_save[i] = output_b
        i += 1

        # mid testing
        answer_b = answer_b.reshape(1, -1)
        b_loss.append(loss_fcn(output_b, answer_b[washout[0]:]).item())

        hidden_layer_b = F.pad(input=hidden_b, pad=(2, 0, 0, 0), mode='constant', value=0)

        # #################### is this needed? ############################################
        # loop back to esn_a
        if feedback:
            auto_ba_recon = model_auto_ba(hidden_layer_b)
            hidden_layer_a = F.pad(input=auto_ba_recon, pad=(1, 0, 0, 0), mode='constant', value=0)

    loss = loss_fcn(output_b_save.to(device), Z_data[washout[0]:]).item()
    print("Mega esn time ended in", time.time() - start, "seconds.")

    return loss, file_data_and_time, esn_loss_a, auto_loss_a, esn_loss_b, auto_loss_b, output_b_save


# loss, file_name, a, b, c, d = Mega_esn(feedback=False)
# load_files('31-08-2021_16-54-48')
