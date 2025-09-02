<h1 align="center">
A Foundation Model for Chest X-ray Interpretation with Grounded Reasoning via Online Reinforcement Learning
</h1>

<p align="center">
  <a href=""><b>[📜 Paper]</b></a> •
  <a href="https://github.com/DeepReasoning/DeepMedix-R1"><b>[🐱 GitHub]</b></a>
  
</p>

<p align="center">
Repo for "<a href="" target="_blank">A Foundation Model for Chest X-ray Interpretation with Grounded Reasoning via Online Reinforcement Learning</a>"
</p>

## 🔥 News

- [2025/02] 🔥🔥🔥 Logical reasoning evaluation study of LLMs is accepted by IEEE TKDE!

## 📖 Introduction

<p align="center">
    <img src="evaluation.png" alt="scaling" width="400">
</p>

In this paper, in-depth evaluations are conducted on logical reasoning tasks, discussing whether LLMs are really good logical reasons.
> - First, the logical reasoning evaluations are organized from deductive, inductive, abductive and mixed-form views. We select fifteen logical reasoning datasets to evaluate on three representative LLMs (i.e., text-davinci-003, ChatGPT and BARD) under both zero-shot and few-shot settings.
> - Second, this paper provides fine-level evaluations on four metrics, covering both objective and subjective views. For problematic cases, extensive error attributions are conducted from two dimensions, forming five error types. It uncovers the logical flaws of LLMs and we provide deep analysis on the results.
> - Third, to achieve a fair and pure benchmark for logical reasoning capability, we propose a dataset with neutral content, covering deductive, inductive and abductive settings.




## 🚀 NeuLR


Considering the current benchmarks may not provide neutral content for fair evaluation, we propose the new dataset ***NeuLR*** to benchmark the neutral-content logical reasoning tasks. In column 1∼3 of the able, we provide the statistics of NeuLR. It contains 3 k samples in total, with ***1 k for deductive reasoning***, ***1 k for inductive reasoning*** and ***1k for abductive reasoning***.





<table>
    <tr>
      <th rowspan="2" align="center">Dataset</th>
      <th rowspan="2" align="center">Num.</th>
      <th rowspan="2" align="center">#Hop</th>
      <th colspan="3" align="center">text-davinci-003</th>
      <th colspan="3" align="center">ChatGPT</th>
      <th colspan="3" align="center">BARD</th>
    </tr>
    <tr>
      <td align="center">0-shot</td>
      <td align="center">1-shot</td>
      <td align="center">COT</td>
      <td align="center">0-shot</td>
      <td align="center">1-shot</td>
      <td align="center">COT</td>
      <td align="center">0-shot</td>
      <td align="center">1-shot</td>
      <td align="center">COT</td>
    </tr>
  <tr>
      <td align="center">NeuLR</td>
    <td align="center">3,000</td>
    <td align="center">1~5</td>
    <td align="center">50.93</td>
    <td align="center">59.17</td>
    <td align="center">67.90</td>
    <td align="center">37.27</td>
    <td align="center">48.13</td>
    <td align="center">48.00</td>
    <td align="center">63.67</td>
    <td align="center">65.07</td>
    <td align="center">66.00</td>
    </tr>
  <tr>
      <td align="center">Deductive</td>
      <td align="center">1,000</td>
      <td align="center">2</td>
      <td align="center">59.00</td>
      <td align="center">69.40</td>
      <td align="center">86.10</td>
      <td align="center">85.20</td>
      <td align="center">69.10</td>
      <td align="center">68.30</td>
      <td align="center">87.40</td>
      <td align="center">93.10</td>
      <td align="center">91.90</td>
    </tr>
  <tr>
      <td align="center">Inductive</td>
      <td align="center">1,000</td>
      <td align="center">3</td>
      <td align="center">86.90</td>
      <td align="center">89.60</td>
      <td align="center">95.60</td>
      <td align="center">15.10</td>
      <td align="center">68.60</td>
      <td align="center">69.60</td>
      <td align="center">96.00</td>
      <td align="center">92.60</td>
      <td align="center">96.30</td>
    </tr>
    <tr>
      <td align="center">Abductive</td>
      <td align="center">1,000</td>
      <td align="center">1~5</td>
      <td align="center">6.90</td>
      <td align="center">18.50</td>
      <td align="center">22.00</td>
      <td align="center">11.50</td>
      <td align="center">6.70</td>
      <td align="center">6.10</td>
      <td align="center">7.60</td>
      <td align="center">9.50</td>
      <td align="center">9.80</td>
    </tr>
</table>






From the results, we have the following observations:
> - First, few-shot prompting and chain-ofthought prompting can both boost the performances of LLMs in most cases. Overall, chain-of-thought helps most to the model accuracy.
> - Second, among the zero-shot results of three LLMs, BARD achieves the best performances on NeuLR while ChatGPT ranks last. The differences of zero-shot settings are significant.
> - Third, from the perspective of different reasoning settings, there exist huge differences in results compared with the previous findings. The performances among the reasoning settings are sorted as ***deductive > abductive > inductive***.





## Citation

If you find it helpful, please kindly cite the paper.

```
@article{DBLP:journals/corr/abs-2306-09841,
  author       = {Fangzhi Xu and
                  Qika Lin and
                  Jiawei Han and
                  Tianzhe Zhao and
                  Jun Liu and
                  Erik Cambria},
  title        = {Are Large Language Models Really Good Logical Reasoners? {A} Comprehensive Evaluation and Beyond},
  journal      = {IEEE Transactions on Knowledge and Data Engineering},
  volume       = {37},
  number       = {4},
  pages        = {1620--1634},
  year         = {2025}
}
```

