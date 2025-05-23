计算机视觉期中pj 任务2
下载官方mmdetection代码仓库后，修改了其中的内容
分别为configs/_base_/datasets/coco_detection.py和configs/_base_/datasets/coco_instance.py（分别用于目标检测和实例分割的训练，替换相应的数据加载部分）
mmdet/datasets/coco.py：将coco数据集的类别替换为voc数据集的类别
mmdet/evaluation/functional/class_names.py：将coco数据集的类别替换为voc数据集的类别
tools/analysis_tools/plot_mAP_curves.py:用于绘制mAP曲线
tools/dataset_converters/pascal_voc_seg.py：用于转换数据格式
剩余的为框架在训练时生成的配置文件
训练与测试示例：
首先进入mmdetection目录：cd mmdetection
训练时首先使用命令行：python tools/train.py configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py --work-dir work_dirs/mask-rcnn_voc
自动生成配置文件后，可用 python tools/train.py work_dirs/mask-rcnn_voc/mask-rcnn_r50_fpn_1x_coco.py --resume继续训练
测试时使用python tools/test.py work_dirs/mask-rcnn_voc/mask-rcnn_r50_fpn_1x_coco.py work_dirs/mask-rcnn_voc/epoch_12.pth --show-dir work_dirs/mask-rcnn_voc/test_show
google drive链接上的三个权重文件分别为mask-rcnn和sparse-rcnn目标检测模型与mask-rcnn实例分割模型
