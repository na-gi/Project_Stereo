import cv2
import numpy as np
import glob

'''
findChessboardCorners 识别并存储棋盘格角点（带可视化）

:param path: 图像文件路径
:param board_size: 棋盘格规格[W, H]
:param criteria: 寻找亚像素角点的参数，默认值0-表示不进行亚像素级角点检测
:param delta: 每张图片识别结果显示的时间间隔，默认值-300

:returns: 
    obj_points, img_points: 棋盘格角点的世界坐标和图像坐标对
    size: 图像尺寸
'''
def findChessboardCorners(path,
                          board_size,
                          criteria = 0,
                          delta = 300):
    # 棋盘格规格（W-行 H-列）= 格数 - 1
    W = board_size[0]
    H = board_size[1]

    objp = np.zeros((W * H, 3), np.float32)
    objp[:, :2] = np.mgrid[0:W, 0:H].T.reshape(-1, 2)

    # 储存棋盘格角点的世界坐标和图像坐标对
    obj_points = []  # 世界坐标系中的三维点
    img_points = []  # 图像平面的二维点

    # 加载path下的所有jpg图片
    images = glob.glob(path)

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
            # 存储点
            obj_points.append(objp)
            if criteria != 0:
                # 执行亚像素级角点检测
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                if [corners2]:
                    img_points.append(corners2)
                else:
                    img_points.append(corners)
            else:
                img_points.append(corners)

            # 将角点绘制在图像上
            cv2.drawChessboardCorners(img, (W, H), corners, ret)
            cv2.namedWindow('findCorners', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('findCorners', size[0], size[1])
            cv2.imshow('findCorners', img)
            cv2.waitKey(delta)
    cv2.destroyAllWindows()
    ## 识别格角点 end
    return obj_points, img_points, size

'''
calibrateCamera 相机标定

:param obj_points: 棋盘格角点的世界坐标
:param img_points: 棋盘格角点的图像坐标

:returns: 
    mtx: 相机内参
    dist: 畸变系数 
    rvecs: 旋转矩阵
    tvecs: 平移矩阵
'''
def calibrateCamera(obj_points, img_points, size):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

    # print('ret: ', ret)
    # print('\nmtx:\n', mtx)
    # print('\ndist:\n', dist)
    # print('\nrvecs:\n', rvecs)
    # print('\ntvecs:\n', tvecs)

    return mtx, dist, rvecs, tvecs

'''
draw_line 立体校正检验----画线

:params: 
    image1, image2: 左右相机图像

:returns: 
    output: 合并画线后的图像
'''
def draw_line(image1, image2):
    # 建立输出图像
    height = max(image1.shape[0], image2.shape[0])
    width = image1.shape[1] + image2.shape[1]

    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:image1.shape[0], 0:image1.shape[1]] = image1
    output[0:image2.shape[0], image1.shape[1]:] = image2

    for k in range(25):
        cv2.line(output, (0, 30 * (k + 1)), (2 * width, 30 * (k + 1)), (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)  # 直线间隔：60

    return output


if __name__ == '__main__':
    ## 初始化 begin
    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    obj_points = []         # 世界坐标系中的三维点
    img_points_l = []       # 左相机图像平面的二维点
    img_points_r = []       # 右相机图像平面的二维点
    ## 初始化 end


    ## 双目标定 begin
    # 左右相机的单目标定
    obj_points, img_points_l, size = findChessboardCorners('left/*.jpg', [9,6], criteria, 100)
    mtx_l, dist_l, rvecs_l, tvecs_l = calibrateCamera(obj_points, img_points_l, size)

    obj_points, img_points_r, size = findChessboardCorners('right/*.jpg', [9,6], criteria, 100)
    mtx_r, dist_r, rvecs_r, tvecs_r = calibrateCamera(obj_points, img_points_r, size)

    # 双目立体矫正及左右相机内参进一步修正
    ret, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(obj_points,
                                                                img_points_l, img_points_r,
                                                                mtx_l, dist_l,
                                                                mtx_r, dist_r,
                                                                size, flags=cv2.CALIB_USE_INTRINSIC_GUESS)

    print('ret: ', ret)
    print('\nmtx_l 左相机的内参:\n', mtx_l)
    print('\ndist_l 左相机的畸变系数:\n', dist_l)
    print('\nmtx_r 右相机的内参:\n', mtx_r)
    print('\ndist_r 右相机的畸变系数:\n', dist_r)
    print('\nR 两相机坐标系转换的旋转矩阵:\n', R)
    print('\nT 两相机坐标系转换的平移矩阵:\n', T)
    print('\nE 本质矩阵:\n', E)
    print('\nF 基础矩阵:\n', F)
    ## 双目标定 end

    ## 校正 begin
    # 计算立体校正的映射矩阵
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(mtx_l, dist_l, mtx_r, dist_r, size, R, T)
    # print(R2*np.linalg.inv(R1))

    left_map1, left_map2 = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, size, cv2.CV_16SC2)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, size, cv2.CV_16SC2)

    frame1 = cv2.imread("left/left01.jpg")
    frame2 = cv2.imread("right/right01.jpg")

    img_rectified_l = cv2.remap(frame1, left_map1, left_map2, cv2.INTER_LINEAR)
    img_rectified_r = cv2.remap(frame2, right_map1, right_map2, cv2.INTER_LINEAR)

    result = draw_line(img_rectified_l, img_rectified_r)
    cv2.imshow('result', result)
    cv2.waitKey(5000)
    ## 校正 end



