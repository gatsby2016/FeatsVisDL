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

    parser.add_argument('-P', '--readImgUrl', type=str, default='../data/cat_dog.png',
                        help='Url of the image for feats map visualization')

    parser.add_argument('-S', '--savePth', type=str, default='../results/featsmapVis/',
                        help='the path to save features maps images')
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
    features_blobs.append(output.squeeze().cpu().numpy())


normMean = [0.485, 0.456, 0.406]
normStd = [0.229, 0.224, 0.225]
preprocess = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize(normMean, normStd)])


def main(net, imgUrl, position):
    for pos in position:
        net._modules[pos].register_forward_hook(hook_features)  # get feature maps

    img = cv2.imread(imgUrl, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    torch_img = preprocess(img).unsqueeze(0).cuda()

    with torch.no_grad():
        prob_output = F.softmax(net(torch_img), dim=1)
        # pred = torch.argmax(prob_output, dim=1).squeeze().cpu().numpy()

    print(imgUrl, ' finished features extraction !')


def showAndsaveMap(featsmap, imgUrl, savepath, layername, Show=True):
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    subpath = os.path.join(savepath, imgUrl.split('/')[-1].split('.')[0])
    if not os.path.exists(subpath):
        os.mkdir(subpath)
    newpath = os.path.join(subpath, layername)
    if not os.path.exists(newpath):
        os.mkdir(newpath)

    print('FeatsMap shape: ', featsmap.shape)
    for ind in range(featsmap.shape[0]):
        # print('Now the channel is: ', ind)
        map = featsmap[ind, :, :]
        map = map / map.max()

        newmap = cv2.applyColorMap(np.uint8(map * 255), cv2.COLORMAP_JET)

        savename = os.path.join(newpath, str(ind) + '_channelMap.png')
        cv2.imwrite(savename, newmap)

        if Show: # change per 100 ms
            cv2.imshow('featsMap', newmap)
            cv2.waitKey(1000)


if __name__ == '__main__':
    features_blobs = []
    layers = ['maxpool', 'layer1', 'layer2', 'layer3', 'layer4']

    Args = GetArgs()

    model = LoadNet(Args.modelPth)  # load model
    # print(model)

    main(model, Args.readImgUrl, position=layers)  # get CAM

    for i in range(len(layers)):
        showAndsaveMap(features_blobs[i], Args.readImgUrl, Args.savePth, layers[i], Show=False)