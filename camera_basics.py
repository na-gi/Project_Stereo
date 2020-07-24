import cv2
import numpy as np
import glob

## 初始化 begin
# 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 棋盘格规格（W-行 H-列）= 格数 - 1
W = 9
H = 6

'''
  假定棋盘正好在x-y平面上，z=0，简化初始化步骤
  设定世界坐标系下棋盘格角点的坐标值(x,y,0)
'''
objp = np.zeros((W * H, 3), np.float32)
objp[:,:2] = np.mgrid[0:W, 0:H].T.reshape(-1, 2)

# 储存棋盘格角点的世界坐标和图像坐标对
obj_points = []     # 世界坐标系中的三维点
img_points = []     # 图像平面的二维点

# 加载left文件夹下的所有jpg图片
images = glob.glob('left/*.jpg')
## 初始化 end

## 识别格角点 begin
for fname in images:
    img = cv2.imread(fname)
    # 转灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 获取图像尺寸
    size = gray.shape[::-1]
    # 找到棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, (W, H), None)
    # 如果能找到足够点，将其存储起来
    if ret == True:
        # 执行亚像素级角点检测
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # 存储点
        obj_points.append(objp)

        if [corners2]:
            img_points.append(corners2)
        else:
            img_points.append(corners)
        # 将角点绘制在图像上
        cv2.drawChessboardCorners(img, (W, H), corners, ret)
        cv2.namedWindow('findCorners', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('findCorners', size[0], size[1])
        cv2.imshow('findCorners', img)
        cv2.waitKey(100)
cv2.destroyAllWindows()
## 识别格角点 end

## 相机标定 begin
'''
  input: 所有图片各自角点的三维、二维坐标，图片尺寸
  output: mtx-相机内参; dist-畸变系数; revcs-旋转矩阵; tvecs-平移矩阵
  每张图片都有自己的旋转和平移矩阵，但是相机内参和畸变系数只有一组。
'''
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

print('ret: ', ret)
print('\nmtx:\n', mtx)
print('\ndist:\n', dist)
print('\nrvecs:\n', rvecs)
print('\ntvecs:\n', tvecs)
## 相机标定 end

## 畸变校正 begin
for fname in images:
    img = cv2.imread(fname)
    h, w = img.shape[:2]
    # 根据自由缩放参数优化摄影机矩阵
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # 裁剪图像，输出纠正畸变以后的图片
    x, y, w1, h1 = roi
    dst = dst[y:y + h1, x:x + w1]
    cv2.imshow('dst', dst)
    cv2.waitKey(100)
    cv2.imwrite('calibresult/' + fname[5:], dst)
cv2.destroyAllWindows()
## 畸变校正 end

## 计算误差 begin
tot_error = 0
for i in range(len(obj_points)):
    # 转换物体点到图像点
    imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(img_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    tot_error += error

print ("\ntotal error: ", tot_error / len(obj_points))
## 计算误差 end
