# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Dict, List
# import sacrebleu
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from rouge import Rouge
from rouge_score import rouge_scorer


from mathruler.grader import extract_boxed_content, grade_answer


def format_reward(predict: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, predict)
    return 1.0 if format_match else 0.0


def accuracy_reward(predict: str, ground_truth: str) -> float:
    answer = extract_boxed_content(predict)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(predicts: List[str], ground_truths: List[str], format_weight: float = 0.1) -> List[Dict[str, float]]:
    scores = []
    for predict, ground_truth in zip(predicts, ground_truths):
        predict = re.sub(r"\s*(<|>|/)\s*", r"\1", predict)  # handle qwen2.5vl-32b format
        format_score = format_reward(predict)
        accuracy_score = accuracy_reward(predict, ground_truth)
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )

    return scores



# 自定义


def coordinate_reward(predict_str: str) -> float:  # bounding box的奖励
    value = 0.0
    think_pattern = r'<think>(.*?)</think>'
    think_matches = re.findall(think_pattern, predict_str, re.DOTALL)

    if not think_matches:
        return value

    text = think_matches[0]
    pattern = r'\[([-+]?\d*\.?\d+(?:,\s*[-+]?\d*\.?\d+){3})\]'
    # pattern = r'\[([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\]'
    matches = re.findall(pattern, text)

    value = 0.0
    match_list = [item.split(',') for item in matches]
    for item in match_list:
        new_item = [float(a.strip()) for a in item]
        # print(new_item)
        if max(new_item) > 512:  # 防止出现超过分辨率范围
            value += -0.2
        else:
            value += 0.05
    # value = 0.05 * len(matches)
    if value > 0.15:  # 设置最大值
        value = 0.15

    return value

def accuracy_xray_reward(predict_str: str, ground_truth: str) -> float:  # xray答案 准确率  blue/acc
    bleu_threshold = 80
    min_bleu_threshold = 10
    answer = extract_boxed_content(predict_str)

    if '|' in ground_truth and len(ground_truth.strip().split()) <= 20:  # 列表类型的
        lable_list = ground_truth.strip().split('|')
        lable_list = [item.strip() for item in lable_list]

        pre_list = predict_str.strip().split('|')
        pre_list = [item.strip() for item in pre_list]
        ture_set = set(lable_list) & set(pre_list)
        precision = len(ture_set) / len(pre_list) if len(pre_list) > 0 else 0
        recall = len(ture_set) / len(lable_list) if len(lable_list) > 0 else 0
        # 计算F1
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return f1


    if len(ground_truth.strip().split()) <= 2:  #
        return 1.0 if grade_answer(answer, ground_truth) else 0.0

    # bleu
    refs = [[ground_truth]]
    sys = [answer]
    bleu_str = str(sacrebleu.corpus_bleu(sys, refs))
    bleu_score = re.search(r'BLEU = (\d+\.\d+)', bleu_str).group(1)
    bleu_score = float(bleu_score)
    if bleu_score > bleu_threshold:
        answer_score = 1.0
    elif bleu_score < min_bleu_threshold:
        answer_score = 0.0
    else:
        answer_score = bleu_score / 100.0
    answer_score = answer_score * 1.5
    return answer_score



def accuracy_xray_reward_new(predict_str: str, ground_truth: str) -> float:  # xray答案 准确率  blue/acc
    bleu_threshold = 0.9
    min_bleu_threshold = 0.1
    answer = extract_boxed_content(predict_str)
    if len(answer) == 0:  # 为空
        answer = ' '

    if '|' in ground_truth and len(ground_truth.strip().split()) <= 20:  # 列表类型的
        lable_list = ground_truth.strip().split('|')
        lable_list = [item.strip().lower() for item in lable_list]

        pre_list = answer.strip().split('|')
        pre_list = [item.strip().lower() for item in pre_list]
        ture_set = set(lable_list) & set(pre_list)
        precision = len(ture_set) / len(pre_list) if len(pre_list) > 0 else 0
        recall = len(ture_set) / len(lable_list) if len(lable_list) > 0 else 0
        # 计算F1
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return f1 * 1.5


    if len(ground_truth.strip().split()) <= 2:
        # return 1.5 if grade_answer(answer, ground_truth) else 0.0
        return 1.5 if answer.lower() == ground_truth.lower() else 0.0

    # bleu
    smoothie = SmoothingFunction().method2
    # bleu_score = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothie)
    bleu1 = sentence_bleu([ground_truth], answer, weights=(1, 0, 0, 0), smoothing_function=smoothie)
    bleu4 = sentence_bleu([ground_truth], answer, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

    # rouge = Rouge()
    # rouge_scores = rouge.get_scores(answer, ground_truth)
    # rouge_l = rouge_scores[0]['rouge-l']['f']
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(answer, ground_truth)
    rouge_l = scores['rouge1'].fmeasure


    bleu_score = (bleu1 + bleu4 + rouge_l) / 3
    # print(bleu1, bleu4, rouge_l)
    answer_score = bleu_score
    # if bleu_score > bleu_threshold:
    #     answer_score = 1.0
    # elif bleu_score < min_bleu_threshold:
    #     answer_score = 0.0
    answer_score = answer_score * 1.5
    return answer_score




def compute_xray_score(predicts: List[str], ground_truths: List[str], format_weight: float = 0.1) -> List[Dict[str, float]]:
    scores = []
    for predict, ground_truth in zip(predicts, ground_truths):
        predict = re.sub(r"\s*(<|>|/)\s*", r"\1", predict)  # handle qwen2.5vl-32b format
        format_score = format_reward(predict)
        accuracy_score = accuracy_xray_reward_new(predict, ground_truth)
        coordinate_score = coordinate_reward(predict)
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score + coordinate_score,
                "format": format_score,
                "accuracy": accuracy_score,
                "coordinate": coordinate_score,
            }
        )

    return scores

