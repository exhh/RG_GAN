from .base_options import BaseOptions


class EvalOptions(BaseOptions):

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='eval', help='train, val, test, etc')
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        parser.add_argument('--train_val_test', type=str, default='test')
        parser.add_argument('--datasetname', type=str, default='liver2D')
        parser.add_argument('--datadir', type=str, default='')
        parser.add_argument('--eval_result_folder', type=str, default='experiments')
        parser.add_argument('--filtering', default=False)
        parser.add_argument('--fix_test', default=True)
        parser.add_argument('--test_all', default=True)
        parser.add_argument('--model_suffix', type=str, default='_GA', help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator.')
        parser.add_argument('--area_sel', type=int, default=5, help='selected area to visualize')
        parser.add_argument('--thd_sel', type=float, default=0.0, help='selected threshold to visualize')

        parser.set_defaults(model='eval')
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser
