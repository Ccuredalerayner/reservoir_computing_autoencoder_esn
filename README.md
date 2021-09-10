# reservoir_computing_autoencoder_esn
A working esn with autoencodeing mega feedback loop implementation

How the network works:
1) Train two separate echo state networks (ESN)s with feedback loops of their hidden internal states.
2) Have two autoencoder learn the feedback loops of each ESN.
3) Swap the encoder portions of each of the autoencoders to essentially produce a feedback loop though the hole network
4) The first autoencoder now encodes the first ESNs hidden state and decodes to the second ESNs hidden state
5) The second autoenocder does the opposite.
6) This is what creates the loop.

Movement of data through the network:
1) Input to network is passed into the first ESN (with hidden state of second esn)
2) Output of first ESN is passed into the second ESN with the hidden state of the first network that has been passed though the first autoencoder
3) Output of the second ESN is saved as output of the hole network and hidden state of the second network is passed through the second autoencoder
4) Replete, input to network is passed to first ESN with decoded hidden stat of the second ESN from the second autoencoder.

Images are shown in images folder to help explain.

Must use the https://github.com/stefanonardo/pytorch-esn for the ESN creation.
There are files on each .py explaining some of the more complicated functions and code.
