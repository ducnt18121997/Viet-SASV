import os
import sys

sys.path.append(".")
import glob
import time
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from cores.dataloader import data_loader
from models.classifier import SpeechClassifier, TrainingConfig
from utils.helper import load_pretrained_path, save_config
from setting import SETTING


def do_train():
    ## Define the data loader
    trainloader = data_loader()
    trainLoader = DataLoader(
        trainloader,
        batch_size=SETTING.BATCH_SIZE,
        shuffle=True,
        num_workers=SETTING.N_CPU,
        drop_last=True,
    )

    ## Search for the exist models
    modelfiles = glob.glob("%s/model_0*.model" % SETTING.MODEL_SAVE_PATH)
    modelfiles.sort()

    ## If initial_model is exist, system will train from the initial_model
    if SETTING.INITIAL_MODEL:
        frontend_config, backend_config = load_pretrained_path(SETTING.INITIAL_MODEL)
        s = SpeechClassifier(
            n_class=classes,
            config=TrainingConfig(),
            frontend_version=SETTING.FRONTEND_VERSION,
            frontend_config=frontend_config,
            backend_version=SETTING.BACKEND_VERSION,
            backend_config=backend_config,
        )
        s.load_parameters(SETTING.INITIAL_MODEL)
        epoch = 1

    ## Otherwise, system will try to start from the saved model&epoch
    elif len(modelfiles) >= 1:
        initial_model = modelfiles[-1]
        frontend_config, backend_config = load_pretrained_path(initial_model)
        epoch = int(os.path.splitext(os.path.basename(initial_model))[0][6:]) + 1
        s = SpeechClassifier(
            n_class=classes,
            config=TrainingConfig(),
            frontend_version=SETTING.FRONTEND_VERSION,
            frontend_config=frontend_config,
            backend_version=SETTING.BACKEND_VERSION,
            backend_config=backend_config,
        )
        s.load_parameters(initial_model)

    ## Otherwise, system will train from scratch
    else:
        epoch = 1
        s = SpeechClassifier(
            n_class=classes,
            config=TrainingConfig(),
            frontend_version=SETTING.FRONTEND_VERSION,
            backend_version=SETTING.BACKEND_VERSION,
        )
    save_config(s, SETTING.MODEL_SAVE_PATH)

    results = []
    score_file = open(SETTING.SCORE_SAVE_PATH, "a+")
    exit()
    while 1:
        ## Add augmentation to dataloader
        if epoch > int(SETTING.MAX_EPOCH) / 2:
            trainLoader.dataset.use_augment = True

        ## Training for one epoch
        loss, lr, acc = s.train_network(epoch=epoch, loader=trainLoader)

        ## Evaluation every [test_step] epochs
        if epoch % SETTING.TEST_STEP == 0:
            s.save_parameters(SETTING.MODEL_SAVE_PATH + "/model_%04d.model" % epoch)
            # results mean EER
            results.append(s.eval_network(eval_list=SETTING.EVAL_LIST)[0])
            print(
                time.strftime("%Y-%m-%d %H:%M:%S"),
                "%d epoch, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%"
                % (epoch, acc, results[-1], min(results)),
            )
            score_file.write(
                "%d epoch, LR %f, LOSS %f, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%\n"
                % (epoch, lr, loss, acc, results[-1], min(results))
            )
            score_file.flush()

        if epoch >= SETTING.MAX_EPOCH:
            quit()

        epoch += 1


def do_evaluation():
    ## Only do evaluation, the initial_model is necessary
    if SETTING.ONLY_EVAL:
        s = SpeechClassifier(
            n_class=classes,
            config=TrainingConfig(),
            frontend_version=SETTING.FRONTEND_VERSION,
            backend_version=SETTING.BACKEND_VERSION,
        )

        print("Model %s loaded from previous state!" % SETTING.INITIAL_MODEL)
        s.load_parameters(SETTING.INITIAL_MODEL)

        if SETTING.COHORT_PATH is not None:
            if os.path.exists(SETTING.COHORT_PATH):
                cohort_feats = [
                    torch.from_numpy(x).to(SETTING.DEVICE)
                    for x in np.load(SETTING.COHORT_PATH)
                ]
            else:
                trainloader = data_loader()
                cohort_feats = s.build_cohort(trainloader.data_list)
                np.save(
                    SETTING.COHORT_PATH,
                    [x.detach().cpu().numpy() for x in cohort_feats],
                )

            bestEER, minDCF = s.eval_network_with_asnorm(
                eval_list=SETTING.EVAL_LIST, cohort_embs=cohort_feats
            )
        else:
            bestEER, minDCF = s.eval_network(eval_list=SETTING.EVAL_LIST)
        print("EER %2.2f%%, minDCF %.4f%%" % (bestEER, minDCF))


if __name__ == "__main__":
    if SETTING.ONLY_EVAL:
        speakers = None
        classes = 1
        do_evaluation()
    else:
        speaker_dir = os.path.join(os.path.dirname(SETTING.TRAIN_LIST), "speakers.json")
        speakers = json.load(open(speaker_dir, "r"))
        classes = len(speakers)
        do_train()
