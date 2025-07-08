class Config(object):
    def __init__(self):

        self.log_epoch = 10
        self.epoch = 120    # 100
        self.epochs_unsupervised = 5
        self.patience = 20  # 20 for labeled ratio = 0.1,  10 for labeled ratio = 0.5
        self.num_class = 6
        
        # model configs
        self.input_channels = 9
        self.backbone_output_channels = 128
        self.projection_output_channels = 32

        self.kernel_size = 8
        self.stride = 1
        self.final_out_channels = 128
        self.hid_dim = 128

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99

        # data parameters
        self.drop_last = True
        self.batch_size = 128
