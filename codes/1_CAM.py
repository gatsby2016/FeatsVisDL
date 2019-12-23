import argparse
import numpy as np
import cv2
import os

import torch
import torch.nn.functional as F
from torchvision import models, transforms


def GetArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-M', '--modelPth', type=str, default='../model/resnet34-333f7ec4.pth',
                        help='the network model path')

    parser.add_argument('-P', '--readImgUrl', type=str, default='../data/spider.png',
                        help='Url of the image for attention map visualization')

    parser.add_argument('-S', '--savePth', type=str, default='../results/CAM/',
                        help='the path to save attention maps and overlapped images')
    arg = parser.parse_args()
    return arg


def LoadNet(modelpath):
    net = models.resnet34(pretrained=False)
    net.load_state_dict(torch.load(modelpath))
    net.eval()
    net.cuda()
    return net


# hook the feature extractor
def hook_features(module, input, output):
    features_blobs[0] = output.squeeze()


normMean = [0.485, 0.456, 0.406]
normStd = [0.229, 0.224, 0.225]
preprocess = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize(normMean, normStd)])


# get the corresponding weight to the predicted class
def GetWeights(network, prediction):
    params = list(network.parameters())
    weights = params[-2]
    return weights[prediction, :]


# generate the CAM map
def GetCAM(featsmap, weights):
    weights = torch.reshape(weights, (-1, 1, 1))
    cam = torch.sum(torch.mul(featsmap, weights), dim=0)
    # cam = F.relu(cam)
    cam = cam - torch.min(cam)
    cam_img = cam / torch.max(cam)

    return cam_img.detach().cpu().numpy()


def main(net, imgUrl, position='layer4'):
    net._modules[position].register_forward_hook(hook_features) # get feature maps
    
    img = cv2.imread(imgUrl, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    torch_img = preprocess(img).unsqueeze(0).cuda()

    with torch.no_grad():
        prob_output = F.softmax(net(torch_img), dim=1)
        pred = torch.argmax(prob_output, dim=1).squeeze().cpu().numpy()

    torch_cls_weights = GetWeights(net, pred)
    CAM = GetCAM(features_blobs[0], torch_cls_weights)
    outCAM = cv2.resize(np.uint8(CAM*255), (W, H))

    print(imgUrl, ' finished generation !')
    return pred, outCAM


# save CAM to savepth
def saveCAM(imgUrl, savepath, pred, CAM):
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    name = imgUrl.split('/')[-1]
    savename = name.split('.')[0] + '_' + str(pred)

    img = cv2.imread(imgUrl, 1)

    attentionmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET)
    overlap = attentionmap * 0.5 + img * 0.5

    cv2.imwrite(os.path.join(savepath, savename + '_CAM.png'), attentionmap)
    cv2.imwrite(os.path.join(savepath, savename + '_overlap.png'), overlap)


if __name__ == '__main__':
    features_blobs = [0]
    Args = GetArgs()

    model = LoadNet(Args.modelPth) # load model
    Prediction, CAMap = main(model, Args.readImgUrl, position='layer4') # get CAM

    saveCAM(Args.readImgUrl, Args.savePth, Prediction, CAMap) # save CAM and overlap
