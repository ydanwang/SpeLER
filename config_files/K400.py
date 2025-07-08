class Config(object):
    def __init__(self):
        self.log_epoch = 10
        self.epoch = 50
        self.epochs_unsupervised = 10
        self.num_class = 400
        self.patience = 20

        # model configs
        self.input_channels = 3
        self.num_point = 18
        self.seg = 20
        self.pool_in=10
        self.layout = 'openpose' 
        self.hid_dim = 64 # 128
        self.output_channels = 128
        self.backbone_output_channels = 128
        self.projection_output_channels = 32
    