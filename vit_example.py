import argparse
import cv2
import numpy as np
import torch
from turbojpeg import TurboJPEG, TJPF_GRAY

from model import VideoModel
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit
from utils import CenterCrop

parser = argparse.ArgumentParser()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


parser.add_argument('--gpus', type=str, required=True, default='0,1,2,3')
parser.add_argument('--lr', type=float, required=True, default=3e-4)
parser.add_argument('--batch_size', type=int, required=True, default=400)
parser.add_argument('--n_class', type=int, required=True, default=1000)
parser.add_argument('--num_workers', type=int, required=True, default=8)
parser.add_argument('--max_epoch', type=int, required=True, default=120)
parser.add_argument('--test', type=str2bool, required=True, default='False')

# load opts
parser.add_argument('--weights', type=str, required=False, default=None)

# save prefix
parser.add_argument('--save_prefix', type=str, required=True, default='checkpoints/lrw-baseline/')

# dataset
parser.add_argument('--dataset', type=str, required=True, default='lrw1000')
parser.add_argument('--border', type=str2bool, required=True, default='False')
parser.add_argument('--mixup', type=str2bool, required=True, default='False')
parser.add_argument('--label_smooth', type=str2bool, required=True, default='False')
parser.add_argument('--se', type=str2bool, required=True, default='False')

parser.add_argument('--use-cuda', action='store_true', default=False,
                    help='Use NVIDIA GPU acceleration')
parser.add_argument(
    '--image-path',
    type=str,
    default='/home/mingwu/workspace_czg/data/LRW/LRW1000_Public_pkl_jpeg/tst/782ee92569a7070003cdd98a0b0a5e14_703_707.pkl',
    help='Input image path')
parser.add_argument('--aug_smooth', action='store_true',
                    help='Apply test time augmentation to smooth the CAM')
parser.add_argument(
    '--eigen_smooth',
    action='store_true',
    help='Reduce noise by taking the first principle componenet'
         'of cam_weights*activations')

parser.add_argument(
    '--method',
    type=str,
    default='gradcam',
    help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')

args = parser.parse_args()
args.use_cuda = args.use_cuda and torch.cuda.is_available()
if args.use_cuda:
    print('Using GPU for acceleration')
else:
    print('Using CPU for computation')


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


if __name__ == '__main__':
    """ python vit_gradcam.py --image-path <path_to_image>
    Example usage of using cam-methods on a VIT network.

    """
    jpeg = TurboJPEG()
    args = parser.parse_args()

    # args = {'use-cuda': True,
    #         'image-path': '/home/mingwu/workspace_czg/data/LRW/LRW1000_Public_pkl_jpeg/tst/782ee92569a7070003cdd98a0b0a5e14_703_707.pkl',
    #         'method': 'gradcam'}
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    video_model = VideoModel(args).cuda()
    checkpoint = torch.load('/home/mingwu/workspace_czg/Two_roads/lrw-1000-baseline/_weight_reshape_model.pt')
    video_model.load_state_dict(checkpoint['video_model'])
    video_model.eval()

    video_model = video_model.cuda()

    target_layers = [video_model.video_cnn.frontend3D[-1]]

    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    args.method = "ablationcam"
    if args.method == "ablationcam":
        cam = methods[args.method](model=video_model,
                                   target_layers=target_layers,
                                   use_cuda=args.use_cuda,
                                   # reshape_transform=reshape_transform,
                                   ablation_layer=AblationLayerVit())
    else:
        cam = methods[args.method](model=video_model,
                                   target_layers=target_layers,
                                   use_cuda=args.use_cuda)
        # reshape_transform=reshape_transform)

    pkl = torch.load(args.image_path)
    video = pkl.get('video')
    video = [jpeg.decode(img, pixel_format=TJPF_GRAY) for img in video]
    video = np.stack(video, 0)
    video = video[:, :, :, 0]

    video = CenterCrop(video, (88, 88))
    input_video = torch.FloatTensor(video)[:, None, ...] / 255.0
    input_video = torch.unsqueeze(input_video, 0).cuda()

    t = 0
    for item in pkl['pinyinlable']:
        if item == 0:
            break
        t += 1
    pkl['target_lengths'] = torch.tensor(t).numpy()

    pinyinlable = np.full((40), 28).astype(pkl['pinyinlable'].dtype)
    t = pkl['pinyinlable'].shape[0]
    pinyinlable[:t, ...] = pkl['pinyinlable'].copy()
    pkl['pinyinlable'] = pinyinlable

    border = pkl['duration']

    # targets = pinyinlable

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    targets = None

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 1
    grayscale_cam = cam(input_tensor=input_video,
                        targets=targets,
                        eigen_smooth=args.eigen_smooth,
                        aug_smooth=args.aug_smooth)

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]

    cam_image = show_cam_on_image(input_video[0], grayscale_cam)
    cv2.imwrite(f'{args.method}_cam.jpg', cam_image)
