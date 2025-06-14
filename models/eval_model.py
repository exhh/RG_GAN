from .base_model import BaseModel
from . import networks
import click
from .models import models
from .models import get_LD_model

class EvalModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        assert not is_train, 'EvalModel cannot be used during training time'
        parser.set_defaults(dataset_mode='test')
        
        parser.add_argument('--model_LD', default='unet2d', type=click.Choice(models.keys()))

        return parser

    def __init__(self, opt):
        assert(not opt.isTrain)
        BaseModel.__init__(self, opt)
        self.loss_names = []
        self.visual_names = ['pred', 'image_paths', 'test_B', 'test_B_label', 'test_B_mask', 'test_B_mask_ori']
        self.model_names = ['LD' + opt.model_suffix]
        self.netLD = get_LD_model(opt.model_LD, finetune=True)

        setattr(self, 'netLD' + opt.model_suffix, self.netLD)

    def set_input(self, input):
        self.image_paths = input['B_test_paths']
        self.test_B = input['B_test'].to(self.device)
        self.test_B_label = input['B_test_label'].to(self.device)
        self.test_B_mask = input['B_test_mask'].to(self.device)
        self.test_B_mask_ori = input['B_test_mask_ori']

    def forward(self):
        self.pred = self.netLD.predict(self.test_B) 

    def optimize_parameters_GD(self):
        pass
    
    def optimize_parameters_LD(self):
        pass
    
    def optimize_parameters_all(self):
        pass
