import base64
import pathlib
import random
import sys
import time

import cv2
import numpy as np

# from matplotlib import pyplot as plt


# マスク画像を用いてgrabCutする
def grabWithImg(mask2, last):
    back = 2
    front = 3
    if last:
        back = 0
        front = 1

    mask[mask2 == 3] = 3  # なんでmask2を参照しているか分からんくなってきた
    # 1つ前のマスクで背景 らしかった 領域を、次のマスクでも同等に扱う
    mask[(mask2 == 2) | (previousMask == 2)] = 2
    mask[newmask2 == 0] = 0  # マウスで描いた白と黒は、完全に前景か、完全に背景として扱う
    mask[newmask2 == 255] = 1
    mask[temp == 0] = 0  # 画像枠周辺の領域は背景として扱う

    cv2.grabCut(img, mask, None, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_MASK)

    mask2 = np.where((mask == 2) | (mask == 0), back, front).astype('uint8')
    mask3 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    ret = img_raw * mask3[:, :, np.newaxis]

    cv2.imshow("ret", ret)
    # 背景を白にする。
    ret_white_background = ret.copy()
    black = [0, 0, 0]
    gray = [100, 100, 100]
    # 画像中の黒色部分([0, 0, 0])をグレー([100, 100, 100])に置換
    ret_white_background[np.where((ret_white_background == black).all(axis=2))] = gray

    return ret, ret_white_background


# mouse callback function
# img_temp windowにマウスのドラッグ操作が行なわれたときに、太さ5の線を引いていく
def draw_line(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Down:" + str(x) + ", " + str(y))
        draw_line.flag = True
        draw_line.befx = x
        draw_line.befy = y
    elif event == cv2.EVENT_LBUTTONUP:
        print("Up:" + str(x) + ", " + str(y))
        draw_line.flag = False
    elif event == cv2.EVENT_MOUSEMOVE and draw_line.flag:
        cv2.line(img_temp_background_white, (draw_line.befx, draw_line.befy), (x, y), color, 5)
        cv2.line(newmask2, (draw_line.befx, draw_line.befy), (x, y), color, 5)
        draw_line.befx = x
        draw_line.befy = y


# 輪郭追跡
def trackContour(img_bin):
    direction = ((-1, 1), (0, 1), (1, 1), (1, 0),
                 (1, -1), (0, -1), (-1, -1), (-1, 0))
    ret = np.zeros(img_bin.shape[:2], np.uint8)

    fin = False
    for y in range(img_bin.shape[0]):  # width
        for x in range(img_bin.shape[1]):  # width
            if img_bin[y][x] == 255:
                fin = True
                break
        if fin:
            break

    # 6 5 4
    # 7 a 3   (入ってきた方向 + 1) % 8 が次の探索index
    # 0 1 2
    inDirection = 7
    ret[y][x] = 128  # start地点を特別に128とする
    fin = False
    cornerCount = 1
    corners = []

    while True:
        cornerCount = cornerCount + 1
        contourGray = 129  # 枠線の色
        if cornerCount == 5:  # 5点ごとにコーナーとして扱う点は255にする
            contourGray = 255
            cornerCount = 0

        for i in range(7):
            # 次に見る画素の方向

            index = (inDirection + i + 1) % 8
            a = direction[index][0]
            b = direction[index][1]

            # 画像外じゃないか
            if (x + a < 0 or x + a >= img_bin.shape[1]) or (y + b < 0 or y + b >= img_bin.shape[0]):
                continue

            # まだ未探索の輪郭だったら輪郭と認識
            if (img_bin[y + b][x + a] == 255):
                if (ret[y + b][x + a] != 128):
                    ret[y + b][x + a] = contourGray
                    x = x + a
                    y = y + b
                    inDirection = (index + 4) % 8
                    break
                else:
                    ret[y + b][x + a] = 255
                    fin = True

        if cornerCount == 0:  # 5回目の時、コーナーとする認識
            corners.append([x, y])

        if fin:
            break

    return ret, corners


# Douglas−Peuckerアルゴリズムによる輪郭の単純化
def cornerReduction(corners):
    distThreshold = 0.8
    cornerTemp = []
    # print(corners)

    for point in range(len(corners)):
        x0 = corners[point][0]
        y0 = corners[point][1]
        x1 = corners[0][0]
        y1 = corners[0][1]
        x2 = corners[-1][0]
        y2 = corners[-1][1]
        # print(x0,y0,x1,y1,x2,y2)
        a = y2 - y1
        b = x2 - x1
        c = y1 * b - x1 * a
        distance = abs(x0 * a - y0 * b + c) / np.sqrt(a * a + b * b)  # 点と直線の距離
        cornerTemp.append(distance)

    dMax = 0
    dMaxIndex = 0
    for index in range(len(cornerTemp)):
        if cornerTemp[index] > dMax:
            dMax = cornerTemp[index]
            dMaxIndex = index

    # print("dMax:" + str(dMax))
    if dMax > distThreshold:
        head = cornerReduction(corners[0:dMaxIndex + 1])
        tail = cornerReduction(corners[dMaxIndex:])
        # print(str(corners[0:dMaxIndex+1]) + str(corners[dMaxIndex:-1]))
        importantCorners = head + tail[1:]
    else:
        # print("delete:" + str(corners[1:-1]))
        del corners[1:-1]  # いらない要素を削除
        importantCorners = corners

    return importantCorners


def outputResults(
        output_path,
        videoTime,
        img_raw,
        img,
        filePath,
        labelName,
        importantCorners,
        rateX,
        rateY,
        smallWidth,
        smallHeight):
    cv2.imwrite("./contourResults/" + videoTime + ".jpg", img_raw)
    cv2.imwrite(f"{output_path}/" + videoTime + ".jpg", img)

    # Base64で画像を文字列化したものをjsonファイルに含める必要あり
    with open(f"{output_path}/" + videoTime + ".jpg", "rb") as f:
        img_base64 = base64.b64encode(f.read())

    with open(filePath, mode='w') as f:  # 衝撃的なほどjson系のライブラリを使えばよかった。。。
        f.write("{\n")
        f.write("\t\"version\": \"3.16.2\",\n")
        f.write("\t\"flags\": {},\n")
        f.write("\t\"shapes\": [\n")
        f.write("\t\t{\n")
        f.write("\t\t\t\"label\": \"" + labelName + "\",\n")
        f.write("\t\t\t\"line_color\": null,\n")
        f.write("\t\t\t\"fill_color\": null,\n")

        f.write("\t\t\t\"points\": [\n")
        for index in range(len(importantCorners) - 1):  # 輪郭の描画
            x1 = importantCorners[index][0]  # 今見ている点
            y1 = importantCorners[index][1]  # 今見ている点
            f.write("\t\t\t\t[" + str(x1 * rateX) + "," + str(y1 * rateY) + "],\n")
        x1 = importantCorners[len(importantCorners) - 1][0]  # 今見ている点
        y1 = importantCorners[len(importantCorners) - 1][1]  # 今見ている点
        f.write("\t\t\t\t[" + str(x1 * rateX) + "," + str(y1 * rateY) + "]\n")
        f.write("\t\t\t],\n")

        f.write("\t\t\t\"shape_type\": \"polygon\",\n")
        f.write("\t\t\t\"flags\": {}\n")
        f.write("\t\t}\n")
        f.write("\t],\n")
        f.write("\t\"lineColor\": [0,255,0,128],\n")
        f.write("\t\"fillColor\": [255,0,0,128],\n")
        f.write("\t\"imagePath\": \"{}/".format(output_path.name) + videoTime + ".jpg\",\n")
        f.write("\t\"imageData\": \"" + (img_base64).decode() + "\",\n")  # 最初に 「b'」 という文字が入らないようにdecode
        f.write("\t\"imageHeight\": " + str(smallHeight) + ",\n")
        f.write("\t\"imageWidth\": " + str(smallWidth) + "\n")
        f.write("}")


print("run!")
args = sys.argv
output_path = pathlib.Path(args[1])

smallWidth = 550  # 前景背景抽出時の画像横のサイズ
microWidth = 550  # 輪郭抽出時の処理が多少重たいのでそのときの画像サイズをここで調節(横サイズ)


# 処理する動画を読み込み、処理に必要な複製等を用意
video_dir = pathlib.Path('./videos')
video_files = [p for p in video_dir.glob('**/*') if p.is_file() and p.name != '.gitkeep']
background_files = [p for p in pathlib.Path('./background').glob('*') if p.is_file() and p.name != '.gitkeep']

for j, video_file in enumerate(video_files):
    video = cv2.VideoCapture(str(video_file))
    labelName = video_file.parent.name
    # ex) labelName = "controller"

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  # 動画の画面横幅
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 動画の画面縦幅
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # 総フレーム数
    frame_rate = video.get(cv2.CAP_PROP_FPS)  # フレームレート(fps)
    color = (255, 255, 255)  # マウスでの描画色

    videoTime = 0
    video.set(cv2.CAP_PROP_POS_FRAMES, int(videoTime * frame_rate))
    ret, img = video.read()

    # 縮小
    smallHeight = int(img.shape[0] * smallWidth / img.shape[1])
    img = cv2.resize(img, (smallWidth, smallHeight))

    img_raw = img.copy()  # 自由に位置を指定できるやつ用のimg
    img2 = img.copy()  # 自由に位置を指定できるやつ用のimg
    temp = img.copy()  # 自由に位置を指定できるやつ用のimg
    img_temp_background_white = img.copy()

    mask = np.zeros(img.shape[:2], np.uint8)
    previousMask = np.zeros(img.shape[:2], np.uint8)
    newmask2 = np.zeros(img.shape[:2], np.uint8)  # マウスでマスク画像を描いていく
    newmask2 = newmask2 + 128

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    while True:
        # video.set(cv2.CAP_PROP_POS_FRAMES, int(videoTime * frame_rate))
        video.set(cv2.CAP_PROP_POS_FRAMES, int((videoTime) * frame_rate))
        ret, img = video.read()
        # 動画終了
        if not ret:
            break

        # jsonファイルの出力準備（既にファイルがあった場合は飛ばす）
        filePath = f"{output_path}/{labelName}_{video_file.stem}_{videoTime}.json"
        try:
            with open(filePath, mode='x') as f:
                print("make file : " + filePath)
        except BaseException:
            print(filePath + " is already done!")
            videoTime += 1
            continue

        img = cv2.resize(img, (smallWidth, smallHeight))
        img_raw = img.copy()
        img2 = img.copy()
        img_temp_background_white = img.copy()

        mask = mask * 0
        newmask2 = np.zeros(img.shape[:2], np.uint8)  # マウスでマスク画像を描いていく
        newmask2 = newmask2 + 128

        # grabCut用の初期化
        draw_line.flag = False
        draw_line.befx = 0
        draw_line.befy = 0

        # ↓ 処理開始 ↓
        # 真ん中の方のみを抽出
        # rect = (int(smallWidth/10), int(smallHeight/10), int(smallWidth*8/10), int(smallHeight*8/10))
        rect = (1, 1, smallWidth - 2, smallHeight - 2)
        cv2.grabCut(img, mask, rect, bgdModel,
                    fgdModel, 3, cv2.GC_INIT_WITH_RECT)

        mask2 = np.where((mask == 2) | (mask == 0), 2, 3).astype('uint8')
        mask3 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        temp = mask.copy()
        mask[previousMask == 2] = 2
        # cv2.imshow("prev", previousMask*50)

        cv2.namedWindow('img_temp')
        cv2.setMouseCallback('img_temp', draw_line)

        img2, img_temp_background_white = grabWithImg(mask2, False)
        img_temp = img2.copy()  # 自由に位置を指定できるやつ用のimg
        img_temp_rotated = img2.copy()  # 自由に位置を指定できるやつ用のimg

        next_video = False
        while(1):
            # mask[(mask == 2)] = 127
            # mask[(mask == 3)] = 200
            # mask[mask == 1] = 255
            cv2.imshow("mask", mask * 85)
            # mask[mask == 255] = 1 # いらない気がする

            cv2.imshow('raw', img_raw)
            cv2.imshow('newmask2', newmask2)
            cv2.imshow('img_temp', img_temp_background_white)

            key = cv2.waitKey(20)
            if key == 119:  # "w"キーを入力で、前景（白）をimg_temp windowに描画できるように
                print("white")
                color = (255, 255, 255)
            elif key == 98:  # "b"
                print("black")
                color = (0, 0, 0)
            elif key == 103:  # "g"キーを入力で、grabCutを実行
                img2, img_temp_background_white = grabWithImg(mask2, False)
                img_temp = img2.copy()  # 自由に位置を指定できるやつ用のimg
            elif key == 27:  # "ESC"キーを入力で、前景と背景のおおよその分離を終了
                # img2, img_temp_high_contrast = grabWithImg(mask2, True)
                break
            elif key == 113:  # "q"キーを入力で、プログラムを途中終了
                next_video = True
                break

        if next_video:
            break
        # cv2.destroyAllWindows()
        previousMask = mask  # 1つ過去のマスク画像を、次のマスク画像の背景らしい領域として利用

        startTime = time.time()

        # 輪郭抽出は時間がかかりそうなので画像を縮小
        microHeight = int(img2.shape[0] * microWidth / img2.shape[1])
        img2_small = cv2.resize(img2, (microWidth, microHeight))

        kernel = np.ones((3, 3), np.uint8)    # 収縮用のカーネル
        kernel2 = np.ones((1, 1), np.uint8)    # 収縮用のカーネル
        img_gray = cv2.cvtColor(img2_small, cv2.COLOR_RGB2GRAY)
        ret, img_bin = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY)

        img_bin = cv2.erode(img_bin, kernel, iterations=3) # Default : iterations=3
        img_bin = cv2.dilate(img_bin, kernel, iterations=1) # Default : iterations=4
        mask_bin = np.where((img_bin == 0), 0, 1).astype('uint8')
        img2_small = img2_small * mask_bin[:, :, np.newaxis]

        # print("before tracking : " + str(time.time() - startTime))

        # 輪郭追跡とコーナーの単純化
        img_contour, importantCorners = trackContour(img_bin)
        importantCorners = cornerReduction(importantCorners)
        # print("\r\nlength:"+str(len(importantCorners)))
        # print("importantCorners:"+str(importantCorners))

        # print("after tracking : " + str(time.time() - startTime))

        # microWidht/Heightを、元のsmallWidth/Heightのサイズに直す比を事前に計算
        rateX = smallWidth / img_contour.shape[1]
        rateY = smallHeight / img_contour.shape[0]
        resized_points = []
        for index in range(len(importantCorners)):  # 輪郭の描画
            x1 = importantCorners[index][0]  # 今見ている点
            y1 = importantCorners[index][1]  # 今見ている点
            x2 = importantCorners[(index + 1) % len(importantCorners)][0]  # 次の点
            y2 = importantCorners[(index + 1) % len(importantCorners)][1]  # 次の点

            img_contour[y1][x1] = 200

            # print(int(x1*rateX),int(y1*rateY))
            cv2.line(img_raw, (int(x1 * rateX), int(y1 * rateY)),
                     (int(x2 * rateX), int(y2 * rateY)), (255, 0, 0), 2)
            cv2.circle(img_raw, (int(x1 * rateX), int(y1 * rateY)),
                       2, (0, 0, 255), -1)
            resized_points.append([int(x1 * rateX), int(y1 * rateY)])

        img2_mask = np.zeros((smallHeight, smallWidth, 3), np.uint8)
        img2_mask = cv2.fillPoly(img2_mask, [np.array(resized_points)], (255, 255, 255))

        # img2_small[img_contour == 129] = (255, 0, 0)
        # img2_small[img_contour == 200] = (0, 255, 0)
        # img2_small[img_contour == 255] = (0, 0, 255)

        # img2_temp = cv2.resize(img2_small, (smallWidth, smallHeight))

        # 出力結果の確認
        cv2.imshow('img_bin', img_bin)
        cv2.imshow('img_contour', img_contour)
        cv2.imshow('img_raw', img_raw)
        # key = cv2.waitKey(10)
        # cv2.destroyAllWindows()

        outputResults(output_path, f"{labelName}_{video_file.stem}_{videoTime}", img_raw, img,
                      filePath, labelName, importantCorners, rateX, rateY, smallWidth, smallHeight)
        # ファイル出力たち

        # 学習精度向上を目指し、データ拡張を行う
        # for count in range(2):

        for count in range(10):
            size = random.uniform(0.6, 1.2)
            degree = random.uniform(-30, 30)
            backImgNum = int(random.uniform(0, len(background_files) - 1))
            centerX = random.random()
            centerY = random.random()

            randomMat = cv2.getRotationMatrix2D(
                (int(smallWidth * centerX), int(smallHeight * centerY)), degree, size)
            print(str(background_files[backImgNum]))
            print(randomMat)

            affine_img = cv2.warpAffine(
                img2, randomMat, (smallWidth, smallHeight))
            affine_mask = cv2.warpAffine(
                img2_mask, randomMat, (smallWidth, smallHeight))
            affine_gray = cv2.cvtColor(affine_mask,cv2.COLOR_RGB2GRAY)
            _, affine_bin = cv2.threshold(affine_gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(affine_gray)
            _, affine_bin = cv2.threshold(affine_gray, 10, 1, cv2.THRESH_BINARY)
            affine_img = affine_img * affine_bin[:, :,np.newaxis]

            backImg = cv2.imread(str(background_files[backImgNum]))
            # 背景画像のHeight
            backH = int(backImg.shape[0] * smallWidth / backImg.shape[1])
            backImg = cv2.resize(backImg, (smallWidth, backH))

            # ランダムに上下左右に対象物の位置を移動させる
            addX = int(random.uniform(-smallWidth / 4, smallWidth / 2))
            addY = int(random.uniform(0, backImg.shape[0] * 2 / 3))

            # コントローラを重畳する
            for y in range(smallHeight):
                if affine_img.shape[0] <= y:
                    continue

                for x in range(smallWidth):
                    if affine_img.shape[1] <= x:
                        continue

                    # backImgに合わせるときにxがはみ出ないか
                    if backImg.shape[1] <= addX + x or addX + x < 0:
                        continue
                    # backImgに合わせるときにxがはみ出ないか
                    elif backImg.shape[0] <= addY + y or addY + y < 0:
                        continue

                    # print(x,y,addX,addY)
                    if affine_img[y][x][0] != 0:  # TODO:out of rangeの発生を防ぐ
                        backImg[addY + y][addX + x] = affine_img[y][x]

            visualizedImg = backImg.copy()
            expandCorners = []

            for index in range(len(importantCorners)):  # 輪郭の描画
                x1 = importantCorners[index][0]  # 今見ている点
                y1 = importantCorners[index][1]  # 今見ている点
                x2 = importantCorners[(index + 1) %
                                      len(importantCorners)][0]  # 次の点
                y2 = importantCorners[(index + 1) %
                                      len(importantCorners)][1]  # 次の点

                # アフィン変換後の座標
                x1_ = x1 * randomMat[0][0] + y1 * \
                    randomMat[0][1] + randomMat[0][2] + addX
                y1_ = x1 * randomMat[1][0] + y1 * \
                    randomMat[1][1] + randomMat[1][2] + addY
                x2_ = x2 * randomMat[0][0] + y2 * \
                    randomMat[0][1] + randomMat[0][2] + addX
                y2_ = x2 * randomMat[1][0] + y2 * \
                    randomMat[1][1] + randomMat[1][2] + addY

                # アフィン変換後に画面外な輪郭を無視
                if affine_img.shape[1] <= x1_ - addX:
                    x1_ = affine_img.shape[1] + addX
                elif x1_ - addX < 0:
                    x1_ = addX
                elif affine_img.shape[0] <= y1_ - addY:
                    y1_ = affine_img.shape[0] + addY
                elif y1_ - addY < 0:
                    y1_ = addY

                # 背景画像外にある輪郭を無視
                if backImg.shape[1] <= x1_:
                    x1_ = backImg.shape[1]
                elif x1_ - addX < 0:
                    x1_ = 0
                elif backImg.shape[0] <= y1_:
                    y1_ = backImg.shape[0]
                elif y1_ < 0:
                    y1_ = 0
                if backImg.shape[1] <= x2_:
                    x2_ = backImg.shape[1]
                elif x2_ < 0:
                    x2_ = 0
                elif backImg.shape[0] <= y2_:
                    y2_ = backImg.shape[0]
                elif y2_ < 0:
                    y2_ = 0

                expandCorners.append([x1_, y1_])

                # 輪郭の視覚化
                cv2.line(visualizedImg, (int(x1_ * rateX), int(y1_ * rateY)),
                         (int(x2_ * rateX), int(y2_ * rateY)), (255, 0, 0), 2)
                cv2.circle(visualizedImg, (int(x1_ * rateX),
                           int(y1_ * rateY)), 2, (0, 0, 255), -1)

            filePath = f"{output_path}/{labelName}_{video_file.stem}_{videoTime}_{count}.json"

            if len(expandCorners) <= 0:
                continue
            outputResults(output_path, f"{labelName}_{video_file.stem}_{videoTime}_{count}", visualizedImg,
                          backImg, filePath, labelName, expandCorners, 1, 1, backImg.shape[1], backImg.shape[0])

        cv2.imshow('back_img', backImg)
        # key = cv2.waitKey()

        print("imwrite time : " + str(time.time() - startTime))

        videoTime += 1

    # cv2.waitKey()
