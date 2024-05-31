# The new config inherits a base config to highlight the necessary modification
_base_ = '../../mmdetection/configs/faster_rcnn/faster-rcnn_r50-caffe_c4-1x_coco.py'

# Clarify the dataset type
dataset_type = 'CocoDataset'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1)))

# Modify dataset related settings
data_root = '../../data/data_japanese_knotweed/'
metainfo = {
    'classes': ('Duizendknoop', ),
    'palette': [
        (220, 20, 60),
    ]
}
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/new_train_annotations_coco.json',
        data_prefix=dict(img='train/')))
val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/new_val_annotations_coco.json',
        data_prefix=dict(img='val/')))
test_dataloader = dict(
    batch_size=8,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/new_test_annotations_coco.json',
        data_prefix=dict(img='test/')))

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + 'annotations/new_val_annotations_coco.json')
test_evaluator = dict(ann_file=data_root + 'annotations/new_test_annotations_coco.json')

# We can use the pre-trained Faster RCNN model to obtain higher performance
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_c4_mstrain_1x_coco/faster_rcnn_r50_caffe_c4_mstrain_1x_coco_20220316_150527-db276fed.pth'

