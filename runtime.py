import utils.utils as utils
from bidir_trainer import BiDirectionalTrainer
import argparse
from PIL import Image
import os
from build_vocab import *

parser = argparse.ArgumentParser()
parser.add_argument("--platform", default='laptop')
parser.add_argument("--type", default='gan',
                    help='GAN archiecture to use '
                         '(gan | stackgan | wgan | vanilla_gan | '
                         'vanilla_wgan | pretrain_img2txt | pretrain_txt2img). '
                         'default = gan. Vanilla mean not conditional')
parser.add_argument("--lr", default=0.0002, type=float)
parser.add_argument("--l1_coef", default=50, type=float)
parser.add_argument("--l2_coef", default=100, type=float)
parser.add_argument("--diter", default=5, type=int,
                    help='Only for WGAN, number of iteration for discriminator for each iteration of the generator. default = 5')
parser.add_argument("--vis_screen", default='gan',
                    help='The visdom env name for visualization. default = gan')
parser.add_argument("--save_path", default='')
parser.add_argument("--inference", default=False, action='store_true')
parser.add_argument('--dataset', default='flowers')
parser.add_argument('--split', default=0, type=int,
                    help='An integer indicating which split to use (0 : train | 1: valid | 2: test). default = 0')
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('-n_worker', '--num_workers', default=8, type=int)
parser.add_argument('-epoch', '--epochs', default=100, type=int)
parser.add_argument('--text_embed_dim', default=128, type=int)
parser.add_argument('--hidden_dim', default=128, type=int)

parser.add_argument("-r", "--resume", default=False, action='store_true', help='resume training by loading optimizer state')
parser.add_argument('--checkpoint', default=None, type=str,
                    help='Path to checkpoint of previous trained models.')

parser.add_argument("-cls", "--cls", default=False, action='store_true',
                    help='Boolean flag to whether train with GAN-CLS (Matching-aware discriminator) algorithms or not. default=False'
                         'see https://hci.iwr.uni-heidelberg.de/system/files/private/downloads/1009852523/frank_gabel_eml2018_report.pdf')
parser.add_argument("-int", "--interp", default=False, action='store_true',
                    help=' Learning with manifold interpolation (GAN-INT)')
parser.add_argument("-uni", "--unimodal_disc", default=False, action='store_true', help='')
parser.add_argument("-deno", "--denoised_reconstruct", default=False, action='store_true', help='')
parser.add_argument("-cycle", "--cyclegan", default=False, action='store_true', help='')
parser.add_argument("-stage", "--image_stagegan", "-stagegan", default=False, action='store_true', help='A multi-stage GAN for image?')

paths = {
    'laptop': 'config_laptop.yaml',
    'desktop': 'config_desktop.yaml',
    'aws': 'config_aws.yaml'
}

args = parser.parse_args()
config = utils.load_config(paths[args.platform])

print('Arguments:')
for k, v in vars(args).items():
    print('\t%s\t:\t%s' % (k, v))

trainer = BiDirectionalTrainer(
    type=args.type,
    config=config,
    dataset=args.dataset,
    split=args.split,
    lr=args.lr,
    save_path=args.save_path,
    l1_coef=args.l1_coef,
    l2_coef=args.l2_coef,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    epochs=args.epochs,
    text_embed_dim=args.text_embed_dim,
    hidden_dim=args.hidden_dim,
    checkpoint=args.checkpoint,
    resume=args.resume,
)

print('is inference?: ' + str(args.inference))
print('model type:' + str(args.type))

if not args.inference and args.type!='cycle_gan':
    # pretrain_img2txt or pretrain_txt2img
    print("Training %s" % args.type)
    trainer.train(args.cls, args.interp,
                  args.unimodal_disc, args.denoised_reconstruct,
                  args.cyclegan, args.image_stagegan)
elif args.inference and args.type=='gan':
    print('gan prediction')
    trainer.predict()