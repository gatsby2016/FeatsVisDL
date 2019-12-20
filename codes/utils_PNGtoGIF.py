# refer to https://blog.csdn.net/sinat_29957455/article/details/85145488

import imageio
from glob import glob
import os

def PNG2GIF(imgPth, savePth):
    gifImg = []
    for path in imgPth:
        gifImg.append(imageio.imread(path))

    imageio.mimsave(savePth, gifImg, fps=3)


if __name__ == '__main__':
    imgpath = '../results/featsmapVis/ball/'
    subpath = os.listdir(imgpath)
    for spath in subpath:
        print(spath)

        imgUrl = glob(os.path.join(imgpath, spath, '*png'))
        imgUrl.sort(key=os.path.getctime)

        if imgUrl == []:
            continue

        if not os.path.exists(imgpath + 'gif/'):
            os.mkdir(imgpath + 'gif/')

        PNG2GIF(imgUrl, imgpath + 'gif/' + spath + '.gif')
