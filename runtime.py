from bidir_trainer import BiDirectionalTrainer
from trainer import Trainer
import argparse
from PIL import Image
import os
from build_vocab import *

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default='bidir',
                    help='basic (general GAN) or bidir (image <-> text)')
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
parser.add_argument("--cls", default=False, action='store_true',
                    help='Boolean flag to whether train with GAN-CLS (Matching-aware discriminator) algorithms or not. default=False'
                         'see https://hci.iwr.uni-heidelberg.de/system/files/private/downloads/1009852523/frank_gabel_eml2018_report.pdf')
parser.add_argument("--interp", default=False, action='store_true',
                    help=' Learning with manifold interpolation (GAN-INT)')
parser.add_argument("--vis_screen", default='gan',
                    help='The visdom env name for visualization. default = gan')
parser.add_argument("--save_path", default='')
parser.add_argument("--inference", default=False, action='store_true')
parser.add_argument('--pre_trained_disc_A', default=None, type=str,
                    help='Discriminator pre-tranined model path used for intializing training.')
parser.add_argument('--pre_trained_gen_A', default=None, type=str,
                    help='Generator pre-tranined model path used for intializing training.')
parser.add_argument('--pre_trained_disc_B', default=None, type=str)
parser.add_argument('--pre_trained_gen_B', default=None, type=str)
parser.add_argument('--dataset', default='flowers')
parser.add_argument('--split', default=0, type=int,
                    help='An integer indicating which split to use (0 : train | 1: valid | 2: test). default = 0')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--epochs', default=55, type=int)
parser.add_argument('--pre_trained_caption_gen', default=None, type=str)
parser.add_argument('--pre_trained_caption_disc', default=None, type=str)
parser.add_argument('--caption_embed_size', default=256, type=int)
parser.add_argument('--caption_hidden_size', default=512, type=int)
parser.add_argument('--caption_num_layers', default=1, type=int)
parser.add_argument('--caption_gen_pretrain_num_epochs', default=100, type=int)
parser.add_argument('--caption_disc_pretrain_num_epochs', default=20, type=int)


args = parser.parse_args()

if args.mode == 'basic':
    trainer = Trainer(
        type=args.type,
        dataset=args.dataset,
        split=args.split,
        lr=args.lr,
        diter=args.diter,
        vis_screen=args.vis_screen,
        save_path=args.save_path,
        l1_coef=args.l1_coef,
        l2_coef=args.l2_coef,
        pre_trained_disc=args.pre_trained_disc_A,
        pre_trained_gen=args.pre_trained_gen_A,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        pre_trained_disc_B=args.pre_trained_disc_B,
        pre_trained_gen_B=args.pre_trained_gen_B,
        pre_trained_caption_gen=args.pre_trained_caption_gen,
        pre_trained_caption_disc=args.pre_trained_caption_disc,
        caption_embed_size=args.caption_embed_size,
        caption_hidden_size=args.caption_hidden_size,
        caption_num_layers=args.caption_num_layers,
        caption_gen_pretrain_num_epochs=args.caption_gen_pretrain_num_epochs,
        caption_disc_pretrain_num_epochs=args.caption_disc_pretrain_num_epochs,
        caption_initial_noise=False
    )
else:
    trainer = BiDirectionalTrainer()

print('is inference?: ' + str(args.inference))
print('model type:' + str(args.type))

if not args.inference and args.type!='cycle_gan':
    # pretrain_img2txt or pretrain_txt2img
    print("Training %s" % args.type)
    trainer.train(args.cls, args.interp)
# elif not args.inference and args.type=='cycle_gan':
#     print('cycle gan')
#     cycle_trainer.train(args.cls)
# elif args.inference and args.type=='cycle_gan':
#     # trainer.predict()
#     print('cycle gan prediction')
#     cycle_trainer.predict()
elif not args.inference and args.type!='cycle_gan':
    print('Training cyclegan')
    trainer.train(args.cls, args.interp)
elif args.inference and args.type=='gan':
    print('gan prediction')
    trainer.predict()
elif args.inference and args.type=='stackgan':
    trainer.predict(args.type)
else:
    print('wrong input...')

