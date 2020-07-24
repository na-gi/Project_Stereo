# Project_Stereo
## 任务
- 第一阶段：单目相机标定与畸变校正
- 第二阶段：双目相机标定，畸变校正与立体校正
- 第三阶段：双目立体匹配与深度估计（未实现）

## 运行环境
- OpenCV 4.1.0
- Python 3.6.6
    - 需要的包：glob，numpy

## 目录结构说明
| 文件名/文件夹名 | 说明 |
| --- | --- |
| calibresult | 左侧相机畸变校正后的图像集（运行camera.py得到） |
| left | 左侧相机拍摄的图像集 |
| right | 右侧相机拍摄的图像集 |
| binocular_basics.py | 完成第二阶段任务 |
| camera_basics.py | 完成第一阶段任务 |