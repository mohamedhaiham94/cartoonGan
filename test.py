from models.models import Generator
from torchvision import utils
import torchvision.utils as vutils

def main():
    
    parser = argparse.ArgumentParser(description='Test Cartoon avatar Gan model')
    parser.add_argument('--model_root', default='pretrained_model/', help='Root directory for models')
    parser.add_argument('--model_name', default='', help='Model G name')
    parser.add_argument('--nz', default=100, type = int , help='Size of generator input')
    opt = parser.parse_args()
    
    model_root = opt.model_root
    model_name = opt.model_name
    nz = opt.nz

    netG = Generator()
    netG.load_state_dict(torch.load(model_root + model_name))

    z = torch.randn(1, nz, 1, 1, device=device)
    fake = netG(z).detach().cpu()
    image_sample = vutils.make_grid(fake, padding=2, normalize=True)
    utils.save_image(image_sample,'output/image_sample.png',padding = 2)

if __name__ == '__main__':
    main()