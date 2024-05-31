# The new config inherits a base config to highlight the necessary modification
_base_ = '../../mmdetection/projects/CO-DETR/configs/codino/co_dino_resize.py'

# Clarify the dataset type
dataset_type = 'CocoDataset'

# We also need to change the num_classes in head to match the dataset's annotation
num_classes = 1

default_scope = 'mmdet'

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
        metainfo=metainfo,
        type=_base_.dataset_type,
        data_root=data_root,
        ann_file='annotations/new_train_annotations_coco.json',
        data_prefix=dict(img='train/')))

val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        metainfo=metainfo,
        type=_base_.dataset_type,
        data_root=data_root,
        ann_file='annotations/new_val_annotations_coco.json',
        data_prefix=dict(img='val/')))

test_dataloader = dict(
    batch_size=4,
    dataset=dict(
        metainfo=metainfo,
        type=_base_.dataset_type,
        data_root=data_root,
        ann_file='annotations/new_test_annotations_coco.json',
        data_prefix=dict(img='test/')))

# test_dataloader = dict(
#     batch_size=4,
#     dataset=dict(
#         metainfo=metainfo,
#         type=_base_.dataset_type,
#         data_root=data_root,
#         ann_file='annotations/november_annotations_coco.json',
#         data_prefix=dict(img='november_test/')))

# test_dataloader = dict(
#     batch_size=4,
#     dataset=dict(
#         metainfo=metainfo,
#         type=_base_.dataset_type,
#         data_root=data_root,
#         ann_file='annotations/mei_annotations_coco.json',
#         data_prefix=dict(img='mei_test/')))


# Modify metric related settings
val_evaluator = dict(ann_file=data_root + 'annotations/new_val_annotations_coco.json')
test_evaluator = dict(ann_file=data_root + 'annotations/new_test_annotations_coco.json')
# test_evaluator = dict(ann_file=data_root + 'annotations/november_annotations_coco.json')
# test_evaluator = dict(ann_file=data_root + 'annotations/mei_annotations_coco.json')

# We can use the pre-trained CO-DETR model to obtain higher performance
# load_from = 'work_dirs/co_detr_pretraining/co_detr_pretrained.pth'
# load_from = 'work_dirs/co_detr_groundlevel/co_detr_groundlevel.pth'
# load_from = 'work_dirs/co_detr_november/co_detr_november.pth'
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_r50_1x_coco-7481f903.pth'
