import torch
from collections import OrderedDict
from torch.backends import cudnn

from .siamese_baseline import SiameseBaselineModelv1, SiameseLocalandMotionModelBIG
from .siamese_baseline_v2 import SiameseLocalandMotionModelBIG_Simple, SiameseLocalandMotionModelBIG_V2
from .siamese_baseline_v2 import SiameseLocalandMotionModelBIG_DualTextCat, SiameseLocalandMotionModelBIG_DualTextAdd
from .siamese_baseline_nl import SiameseLocalandMotionModelBIG_V2_View, \
    SiameseLocalandMotionModelBIG_DualTextAdd_view, SiameseLocalandMotionModelBIG_DualTextCat_view
from .siamese_baseline_v3 import SiameseLocalandMotionModelBIGV3, SiameseLocalandMotionModelBIG_DualTextCat_Multiframes, \
    SiameseLocalandMotionModelBIG_DualTextCat_view_MultiCrops, SiameseLocalandMotionModelBIG_MultiQueries

def build_model(cfg, args):
    if cfg.MODEL.NAME == "base":
        model = SiameseBaselineModelv1(cfg.MODEL)
    elif cfg.MODEL.NAME == "dual-stream":
        model = SiameseLocalandMotionModelBIG(cfg.MODEL)
    elif cfg.MODEL.NAME == 'dual-simple':
        model = SiameseLocalandMotionModelBIG_Simple(cfg.MODEL)
    elif cfg.MODEL.NAME == 'dual-stream-v2':
        model = SiameseLocalandMotionModelBIG_V2(cfg.MODEL)
    elif cfg.MODEL.NAME == 'dual-text-cat':
        model = SiameseLocalandMotionModelBIG_DualTextCat(cfg.MODEL)
    elif cfg.MODEL.NAME == 'dual-text-add':
        model = SiameseLocalandMotionModelBIG_DualTextCat(cfg.MODEL)
    elif cfg.MODEL.NAME == 'dual-stream-v2-view':
        model = SiameseLocalandMotionModelBIG_V2_View(cfg.MODEL)
    elif cfg.MODEL.NAME == 'dual-text-cat-view':
        model = SiameseLocalandMotionModelBIG_DualTextCat_view(cfg.MODEL)
    elif cfg.MODEL.NAME == 'dual-stream-vit':
        model = SiameseLocalandMotionModelBIGV3(cfg.MODEL)
    elif cfg.MODEL.NAME == 'dual-text-cat-multiframes':
        model = SiameseLocalandMotionModelBIG_DualTextCat_Multiframes(cfg.MODEL)
    elif cfg.MODEL.NAME == 'dual-text-cat-view-multi-crops':
        model = SiameseLocalandMotionModelBIG_DualTextCat_view_MultiCrops(cfg.MODEL)
    elif cfg.MODEL.NAME == "dual-stream-multi-queries":
        model = SiameseLocalandMotionModelBIG_MultiQueries(cfg.MODEL)
    else:
        assert cfg.MODEL.NAME in ["base", "dual-stream", "dual-simple", "dual-stream-v2"], f"unsupported model {cfg.MODEL.NAME}"

    ossSaver = args.ossSaver

    if args.resume:
        if cfg.TEST.RESTORE_FROM == "" or cfg.TEST.RESTORE_FROM is None:
            cfg.TEST.RESTORE_FROM = args.logs_dir + "/checkpoint_best_eval.pth"
            if cfg.DATA.USE_OSS:
                cfg.TEST.RESTORE_FROM = ossSaver.get_s3_path(cfg.TEST.RESTORE_FROM)

        checkpoint = ossSaver.load_ckpt(cfg.TEST.RESTORE_FROM)
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:]  # remove `module.`
            if cfg.MODEL.NAME == 'dual-text-cat-multiframes' and "vis_fc_merge" in name:
                pass
            else: 
                #new_state_dict[name] = v
                model.state_dict()[name].copy_(v)
    else:
        print(f"====> load checkpoint from default")

    if args.use_cuda:
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = cfg.TRAIN.BENCHMARK

    return model
