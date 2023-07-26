_base_ = [
    '../_base_/models/segformer_mit-b2_segformer_head_unet_fc.py',
    '../_base_/datasets/ade20k151.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'

]
norm_cfg = dict(type='SyncBN', requires_grad=True)
checkpoint = 'work_dirs/segformer_mit_b2_segformer_head_unet_fc_small_single_step_ade_pretrained_freeze_embed_80k_ade20k151/latest.pth'
# model settings
model = dict(
    type='EncoderDecoderDiffusion',
    freeze_parameters=['backbone', 'decode_head'],
    pretrained=checkpoint,
    backbone=dict(
        type='MixVisionTransformerCustomInitWeights'
    ),
    decode_head=dict(
        _delete_=True,
        type='SegformerHeadUnetFCHeadMultiStep',
        # unet params
        # pretrained=checkpoint,
        dim=256,
        out_dim=256,
        unet_channels=272,
        dim_mults=[1,1,1],
        cat_embedding_dim=16,
        diffusion_timesteps=20,
        inference_timesteps=2,
        # collect_timesteps=[19,18,17,16,15,10,5,0],
        collect_timesteps=[i for i in range(0,20,10)]+[19],
        guidance_scale=.4,
        # decode head params
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=151,
        norm_cfg=norm_cfg,
        align_corners=False,
        ignore_index=0,  # ignore background
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0)
    )
)

optimizer = dict(_delete_=True, type='AdamW', lr=1.5e-4, betas=[0.9, 0.96], weight_decay=0.045)
lr_config = dict(_delete_=True, policy='step',
                 warmup='linear',
                 warmup_iters=1000,
                 warmup_ratio=1e-6,
                 step=20000, gamma=0.5, min_lr=1.0e-6, by_epoch=False)

checkpoint_config = dict(by_epoch=False, interval=16000, max_keep_ckpts=1)

custom_hooks = [dict(
        type='ConstantMomentumEMAHook',
        momentum=0.01,
        interval=25,
        eval_interval=16000,
        auto_resume=True,
        priority=49)
]
