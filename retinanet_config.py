# The new config inherits a base config to highlight the necessary modification
_base_ = '../../mmdetection/configs/retinanet/retinanet_r50_fpn_1x_coco.py'

# Clarify the dataset type
dataset_type = 'CocoDataset'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(    
        bbox_head=dict(num_classes=1))

# Modify dataset related settings
data_root = '../../data/data_japanese_knotweed/'
metainfo = {
    'classes': ('Duizendknoop', ),
    'palette': [
        (220, 20, 60),
    ]
}
train_dataloader = dict(
    batch_size=8,
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

# We can use the pre-trained RetinaNet model to obtain higher performance
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/fp16/retinanet_r50_fpn_fp16_1x_coco/retinanet_r50_fpn_fp16_1x_coco_20200702-0dbfb212.pth'

# Change Optimizer to AdamW
optimizer = dict(type="AdamW", lr=1e-5, weight_decay=0.001)
