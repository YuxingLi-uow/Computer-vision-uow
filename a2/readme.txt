RUN TASK 1:
python siftImages.py D:\Others\CSCI935\a2\img01.jpg


RUN TASK 2:
python siftImages.py D:\Others\CSCI935\a2\img01.jpg D:\Others\CSCI935\a2\img02.jpg D:\Others\CSCI935\a2\img03.jpg D:\Others\CSCI935\a2\img04.jpg


把文件名改成正确的文件名，cmd运行


存在的问题：

0. 用YUV 还是 YCrCb??? pdf说的是 luminance Y component，亮度 Y。（Done YUV in script）

1. task1 中用 + 画出 keypoints？？？未实现 （done，sb要求）
2. task2 中，最后计算 Chi-Square distance，应该用 bag of word 的 histogram 还是用 image 的 histogram 计算距离？
   BOW 的 dissimilarity 结果比较小，<5。 Image 的 histogram 计算的 dissimilarity 很大，>10000。
   pdf中给的 dissimilarity matrix 值都是小于1的？ (done BOW histogram in script)
3. task2 中，计算的 dissimilarity 的矩阵，Image1 vs Image2 和 Image2 vs Image1 的dissimilarity 结果不一致，要计算吗？pdf中没有写明。(done both in script)
   (也就是说矩阵的左下三角要不要计算？)
