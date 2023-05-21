import os
import logging
import math
import numpy as np
import torch.optim as optim
import torch.nn as nn

from models.model import get_model
from utils import get_accuracy, eval_domain_dict
from conf import cfg, load_cfg_fom_args, get_num_classes, get_domain_sequence
from datasets.data_loading import get_source_loader, get_test_loader

from methods.source_only_clip import SourceOnlyCLIP
from methods.bn import AlphaBatchNorm, EMABatchNorm
from methods.tent import Tent
from methods.ttaug import TTAug
from methods.memo import MEMO
from methods.cotta import CoTTA
from methods.gtta import GTTA
from methods.adacontrast import AdaContrast
from methods.rmt import RMT
from methods.eata import EATA
from methods.norm import Norm
from methods.lame import LAME
from methods.sar import SAR, SAM
from methods.rotta import RoTTA


logger = logging.getLogger(__name__)


def evaluate(description):
    load_cfg_fom_args(description)
    assert cfg.SETTING in ["reset_each_shift",           # reset the model state after the adaptation to a domain
                           "continual",                  # train on sequence of domain shifts without knowing when shift occurs
                           "gradual",                    # sequence of gradually increasing / decreasing domain shifts
                           "mixed_domains",              # consecutive test samples are likely to originate from different domains
                           "correlated",                 # sorted by class label
                           "mixed_domains_correlated",   # mixed domains + sorted by class label
                           "gradual_correlated",         # gradual domain shifts + sorted by class label
                           "reset_each_shift_correlated"
                           ]

    num_classes = get_num_classes(dataset_name=cfg.CORRUPTION.DATASET)

    logger.info(f"Setting up test-time adaptation method: {cfg.MODEL.ADAPTATION.upper()}")
    if cfg.MODEL.ADAPTATION == "source":  # BN--0
        model = setup_source()
    elif cfg.MODEL.ADAPTATION == "rmt":
        model = setup_rmt(num_classes)
    else:
        raise ValueError(f"Adaptation method '{cfg.MODEL.ADAPTATION}' is not supported!")

    # get the test sequence containing the corruptions or domain names
    if cfg.CORRUPTION.DATASET in {"domainnet126"}:
        # extract the domain sequence for a specific checkpoint.
        dom_names_all = get_domain_sequence(ckpt_path=cfg.CKPT_PATH)
    elif cfg.CORRUPTION.DATASET in {"imagenet_d", "imagenet_d109"} and not cfg.CORRUPTION.TYPE[0]:
        # dom_names_all = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
        dom_names_all = ["clipart", "infograph", "painting", "real", "sketch"]
    else:
        dom_names_all = cfg.CORRUPTION.TYPE
    logger.info(f"Using the following domain sequence: {dom_names_all}")

    # prevent iterating multiple times over the same data in the mixed_domains setting
    dom_names_loop = ["mixed"] if "mixed_domains" in cfg.SETTING else dom_names_all

    # setup the severities for the gradual setting
    if "gradual" in cfg.SETTING and cfg.CORRUPTION.DATASET in {"cifar10_c", "cifar100_c", "imagenet_c"} and len(cfg.CORRUPTION.SEVERITY) == 1:
        severities = [1, 2, 3, 4, 5, 4, 3, 2, 1]
        logger.info(f"Using the following severity sequence for each domain: {severities}")
    else:
        severities = cfg.CORRUPTION.SEVERITY

    errs = []
    errs_5 = []
    domain_dict = {}

    # start evaluation
    for i_dom, domain_name in enumerate(dom_names_loop):
        # if i_dom == 0 or "reset_each_shift" in cfg.SETTING:
        #     try:
        #         # pre-trained modelの重みをロード
                # model.reset()
        #         logger.info("resetting model")
        #     except:
        #         logger.warning("not resetting model")
        # else:
        #     logger.warning("not resetting model")

        for severity in severities:
            logger.info(f"----- get_test_loader -----")
            test_data_loader = get_test_loader(setting=cfg.SETTING,
                                               adaptation=cfg.MODEL.ADAPTATION,
                                               dataset_name=cfg.CORRUPTION.DATASET,
                                               root_dir=cfg.DATA_DIR,
                                               domain_name=domain_name,
                                               severity=severity,
                                               num_examples=cfg.CORRUPTION.NUM_EX,
                                               domain_names_all=dom_names_all,
                                               alpha_dirichlet=cfg.TEST.ALPHA_DIRICHLET,
                                               batch_size=cfg.TEST.BATCH_SIZE,
                                               shuffle=False,
                                               workers=min(cfg.TEST.NUM_WORKERS, os.cpu_count()))

            logger.info(f"----- get_accuracy -----")
            acc, domain_dict = get_accuracy(
                model, data_loader=test_data_loader, dataset_name=cfg.CORRUPTION.DATASET,
                domain_name=domain_name, setting=cfg.SETTING, domain_dict=domain_dict)

            err = 1. - acc
            errs.append(err)
            if severity == 5 and domain_name != "none":
                errs_5.append(err)

            logger.info(f"{cfg.CORRUPTION.DATASET} error % [{domain_name}{severity}][#samples={len(test_data_loader.dataset)}]: {err:.2%}")

    if len(errs_5) > 0:
        logger.info(f"mean error: {np.mean(errs):.2%}, mean error at 5: {np.mean(errs_5):.2%}")
    else:
        logger.info(f"mean error: {np.mean(errs):.2%}")

    if "mixed_domains" in cfg.SETTING:
        # print detailed results for each domain
        eval_domain_dict(domain_dict, domain_seq=cfg.CORRUPTION.TYPE)


def setup_source():
    """Set up BN--0 which uses the source model without any adaptation."""
    model = SourceOnlyCLIP()
    model.eval()
    return model


def setup_rmt(num_classes):
    # ここでいうmodel: pre-trained model(e.g. ViT)
    # model = RMT.configure_model(model)
    # params, param_names = RMT.collect_params(model)
    # optimizer = setup_optimizer(params)
    
    batch_size_src = cfg.TEST.BATCH_SIZE if cfg.TEST.BATCH_SIZE > 1 else cfg.TEST.WINDOW_LENGTH
    _, src_loader = get_source_loader(dataset_name=cfg.CORRUPTION.DATASET,
                                      root_dir=cfg.DATA_DIR, adaptation=cfg.MODEL.ADAPTATION,
                                      batch_size=batch_size_src, ckpt_path=cfg.CKPT_PATH, percentage=cfg.SOURCE.PERCENTAGE,
                                      workers=min(cfg.SOURCE.NUM_WORKERS, os.cpu_count()))
    rmt_model = RMT(
                    steps=cfg.OPTIM.STEPS,
                    episodic=cfg.MODEL.EPISODIC,
                    window_length=cfg.TEST.WINDOW_LENGTH,
                    dataset_name=cfg.CORRUPTION.DATASET,
                    arch_name=cfg.MODEL.ARCH,
                    num_classes=num_classes,
                    src_loader=src_loader,
                    ckpt_dir=cfg.CKPT_DIR,
                    ckpt_path=cfg.CKPT_PATH,
                    contrast_mode=cfg.CONTRAST.MODE,
                    temperature=cfg.CONTRAST.TEMPERATURE,
                    projection_dim=cfg.CONTRAST.PROJECTION_DIM,
                    lambda_ce_src=cfg.RMT.LAMBDA_CE_SRC,
                    lambda_ce_trg=cfg.RMT.LAMBDA_CE_TRG,
                    lambda_cont=cfg.RMT.LAMBDA_CONT,
                    m_teacher_momentum=cfg.M_TEACHER.MOMENTUM,
                    num_samples_warm_up=cfg.RMT.NUM_SAMPLES_WARM_UP,
                    save_dir = cfg.SAVE_DIR)
    # return rmt_model, param_names
    return rmt_model


def setup_optimizer(params):
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                          lr=cfg.OPTIM.LR,
                          betas=(cfg.OPTIM.BETA, 0.999),
                          weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(params,
                         lr=cfg.OPTIM.LR,
                         momentum=cfg.OPTIM.MOMENTUM,
                         dampening=cfg.OPTIM.DAMPENING,
                         weight_decay=cfg.OPTIM.WD,
                         nesterov=cfg.OPTIM.NESTEROV)
    else:
        raise NotImplementedError


def setup_adacontrast_optimizer(model):
    backbone_params, extra_params = (
        model.src_model.get_params()
        if hasattr(model, "src_model")
        else model.get_params()
    )

    if cfg.OPTIM.METHOD == "SGD":
        optimizer = optim.SGD(
            [
                {
                    "params": backbone_params,
                    "lr": cfg.OPTIM.LR,
                    "momentum": cfg.OPTIM.MOMENTUM,
                    "weight_decay": cfg.OPTIM.WD,
                    "nesterov": cfg.OPTIM.NESTEROV,
                },
                {
                    "params": extra_params,
                    "lr": cfg.OPTIM.LR * 10,
                    "momentum": cfg.OPTIM.MOMENTUM,
                    "weight_decay": cfg.OPTIM.WD,
                    "nesterov": cfg.OPTIM.NESTEROV,
                },
            ]
        )
    else:
        raise NotImplementedError(f"{cfg.OPTIM.METHOD} not implemented.")

    for param_group in optimizer.param_groups:
        param_group["lr0"] = param_group["lr"]  # snapshot of the initial lr

    return optimizer


if __name__ == '__main__':
    evaluate('"Evaluation.')

