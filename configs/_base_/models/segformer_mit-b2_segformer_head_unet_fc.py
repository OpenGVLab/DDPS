# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth'  # noqa
model = dict(
    type='EncoderDecoderFreeze',
    freeze_parameters=['backbone', 'decode_head'],
    pretrained=checkpoint,
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 4, 6, 3],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1),
    decode_head=dict(
        type='SegformerHeadUnetFCHead',
        # unet params
        pretrained='pretrained/segformer_mit-b2_512x512_160k_ade20k_decode_head.pth',
        dim=256,
        out_dim=256,
        unet_channels=272,
        dim_mults=[1,2,4],
        cat_embedding_dim=16,
        # decode head params
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        # ignore_index=0,  # ignore background
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0)
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
