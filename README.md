# annotateFromVideo
これはMask R-CNNやYOLACTなどのInstance segmentationの、訓練データ生成用のアノテーションツールです。
It is an annotation tool for create training data of Instance segmentation, such as Mask R-CNN or YOLACT


[https://youtu.be/Z1RKNU1CqGM:embed:cite]


<br>  
私の環境は下記の通りですが、違っていても動くと思います。
 
* Python : 3.8
* numpy : 1.16
* opencv-python : 4.1.1


<br>  
実行するためのコマンドは下記の通りです。

```
python annotateFromVideo.py targetVideo.mp4 targetObjectName
```

targetVideo.mp4には、Annotationしたい動画のファイル名を、  
targetObjectNameには、対象物体の名称を記述してください。


<br>  
実行中は下記のことができます。

1. [ b ]必ず<b>背景</b>としたい箇所をマークできるモードに変更
1. [ w ]必ず<b>前景</b>としたい箇所をマークできるモードに変更
1. [ g ]背景と前景の抽出を試行する
1. [esc]背景と前景の抽出を確定し、次の画像へ進む
1. [ q ]途中でツールを終了できる

<br>  
～注意点～

* アノテーションをやり直したい場合、その画像のjsonファイルを削除し、プログラムを実行しなおすと再挑戦できます。
* outputResultsなどのフォルダがない場合、上手く実行できない可能性があります。
* データ拡張の枚数を増やしたい場合は、394行目を変更してください。