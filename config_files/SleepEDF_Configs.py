class Config(object):
    def __init__(self):
        self.log_epoch = 10 
        self.epoch = 120 # labeled ratio 0.1: 20, 0.5: 120
        self.epochs_unsupervised = 5 # labeled ratio 0.1: 10, 0.5: 10
        self.num_class = 5
        
        self.patience = 20 # 20 # labeled ratio 0.1: 20, 0.5: 20

        # model configs
        self.backbone_output_channels = 128
        self.projection_output_channels = 32
        self.kernel_size = 25

        self.input_channels = 1
        self.output_channels = 128
        self.hid_dim = 128

        # training configs

        # optimizer parameters
        self.optimizer = 'adam'
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4

        # data parameters
        self.drop_last = True
        self.batch_size = 128