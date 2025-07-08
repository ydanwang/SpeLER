class Config(object):
    def __init__(self):
        self.log_epoch = 10
        self.epoch =  120  # 150   #
        self.epochs_unsupervised = 5 #  10
        self.num_class = 2

        self.patience = 20  # 30 

        # model configs
        self.input_channels = 1
        self.kernel_size = 8
        self.stride = 1
        self.hid_dim = 32

        self.dropout = 0.35
        self.features_len = 24

        # training configs
        
        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
