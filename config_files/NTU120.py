class Config(object):
    def __init__(self):
        self.log_epoch = 10
        self.num_class = 120
        self.patience = 20

        # model configs
        self.input_channels = 3
        self.num_point = 25
        self.num_person = 2
        self.pool_in = 10
        self.seg=20
        self.layout = 'nturgb+d'
        self.kernel_size = 8
        self.hid_dim = 128
        self.output_channels = 128
        self.backbone_output_channels = 128
        self.projection_output_channels = 32
