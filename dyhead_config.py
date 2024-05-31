# The new config inherits a base config to highlight the necessary modification
_base_ = '../../mmdetection/configs/dyhead/atss_r50_fpn_dyhead_1x_coco.py'

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
# test_dataloader = dict(
#     batch_size=8,
#     dataset=dict(
#         data_root=data_root,
#         metainfo=metainfo,
#         ann_file='annotations/november_annotations_coco.json',
#         data_prefix=dict(img='november_test/')))
# test_dataloader = dict(
#     batch_size=8,
#     dataset=dict(
#         data_root=data_root,
#         metainfo=metainfo,
#         ann_file='annotations/mei_annotations_coco.json',
#         data_prefix=dict(img='mei_test/')))

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + 'annotations/new_val_annotations_coco.json')
test_evaluator = dict(ann_file=data_root + 'annotations/new_test_annotations_coco.json')
# test_evaluator = dict(ann_file=data_root + 'annotations/november_annotations_coco.json')
# test_evaluator = dict(ann_file=data_root + 'annotations/mei_annotations_coco.json')


# We can use the pre-trained DyHead model to obtain higher performance
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/dyhead/atss_r50_fpn_dyhead_4x4_1x_coco/atss_r50_fpn_dyhead_4x4_1x_coco_20211219_023314-eaa620c6.pth'
# load_from = 'work_dirs/dyhead_pretraining/dyhead_pretrained.pth'
# load_from = 'work_dirs/dyhead_groundlevel/dyhead_groundlevel.pth'
# load_from = 'work_dirs/dyhead_november/dyhead_november.pth'
