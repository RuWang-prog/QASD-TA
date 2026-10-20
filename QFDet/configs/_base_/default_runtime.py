checkpoint_config = dict(interval=1)   #cx
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None          #cx
resume_from = None    #cx
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded #禁用opencv多线程以避免系统过载
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training 将多进程启动方法设置为“fork”以加快训练速度
mp_start_method = 'fork'

# Default setting for scaling LR automatically 自动缩放LR的默认设置
#   - `enable` means enable scaling LR automatically ，enable表示自动启用缩放LR
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
