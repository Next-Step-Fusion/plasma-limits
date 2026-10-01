This directory contains the EPED neural network used for SPARC studies, trained by Jo Hall (MIT)

This is an ensemble of 10 feed-forward NNs, each with 9 inputs, 2 outputs, and 2 hidden layers of 32 neurons (ReLU activation)

The minimum and maximum of the training dataset input ranges used to generate these models are below, in order of input position:

Ip:     [  1.6  , 14.3   ]
Bt:     [  7.2  , 12.2   ]
R:      [  1.85 ,  1.85  ]
a:      [  0.57 ,  0.57  ]
kappa:  [  1.53 ,  2.29  ]
delta:  [  0.39 ,  0.59  ]
neped:  [  2.84 , 90.235 ]
betan:  [  0.8  ,  1.6   ]
zeff:   [  1.3  ,  2.5   ]

Both kappa and delta were taken from the normalized poloidal flux surface (psi) = 0.995 for the training dataset

A simple test of the NN is provided in "test_script.py" inside this directory, from which an example code for loading and evaluating the network can also be taken

Please cite [M. Muraca et al. 2025 Nucl. Fusion 65 096010] (https://doi.org/10.1088/1741-4326/adf656) in any works using this model
