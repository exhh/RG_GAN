from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import glob
import os


class SingleDataset(BaseDataset):

    def __init__(self, opt):

        BaseDataset.__init__(self, opt)

        self.data_dir = 'lesion_detection/dotatate'
        self.base_data = 'based_folder'

        self.A_paths = sorted(glob.glob(os.path.join(self.data_dir, self.base_data[1:], 'DOTA*/images/*.png')))

        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.transform = get_transform(opt, grayscale=(input_nc == 1), normalize=True)

    def __getitem__(self, index):

        A_path = self.A_paths[index]

        A_img = Image.open(A_path).convert('L') #$
        A = self.transform(A_img)

        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        return len(self.A_paths)
