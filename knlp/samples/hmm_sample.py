# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: hmm_samples
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-03-27
# Description: 实现了使用hmm进行训练并将模型保存在目录中的功能
# -----------------------------------------------------------------------#
from knlp.common.constant import KNLP_PATH
from knlp.seq_labeling.hmm.inference import Inference
from knlp.seq_labeling.hmm.train import Train

# init trainer and inferencer
hmm_inferencer = Inference()
hmm_trainer = Train()


def hmm_train(vocab_set_path, training_data_path, model_save_path):
    """
    This function call hmm trainer and inference. You could just prepare training data and test data to build your own
    model from scratch.

    Args:
        vocab_set_path:
        training_data_path:
        model_save_path:

    Returns:

    """
    hmm_trainer.init_variable(vocab_set_path=vocab_set_path, training_data_path=training_data_path)
    hmm_trainer.build_model(state_set_save_path=model_save_path, transition_pro_save_path=model_save_path,
                            emission_pro_save_path=model_save_path,
                            init_state_set_save_path=model_save_path)
    print(
        "Congratulations! You have completed the training of hmm model for yourself. "
        f"Your training info: vocab_set_path: {vocab_set_path}, training_data_path: {training_data_path}. "
        f"model_save_path: {model_save_path}"
    )


def hmm_inference_init(model_save_path):
    hmm_inferencer.load_mode(state_set_save_path=model_save_path, transition_pro_save_path=model_save_path,
                             emission_pro_save_path=model_save_path,
                             init_state_set_save_path=model_save_path)


def test(sentence):
    return list(hmm_inferencer.cut(sentence))


if __name__ == '__main__':
    vocab_set_path = KNLP_PATH + "/knlp/data/seg_data/train/pku_vocab.txt"
    training_data_path = KNLP_PATH + "/knlp/data/seg_data/train/pku_hmm_training_data_sample.txt"
    model_save_path = KNLP_PATH + "/knlp/model/hmm/"
    hmm_train(vocab_set_path=vocab_set_path, training_data_path=training_data_path, model_save_path=model_save_path)
    hmm_inference_init(model_save_path=model_save_path)
    print(test("大家好，我是你们的好朋友"))
