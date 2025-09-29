# Multimodal Prototype-Based Networks for Interpretable Food Classification (Food-101)

> Novel, multimodal ProtoPNet-style PBN using VisualBERT + generated captions (BLIP/BLIP2) to improve Food-101 classification accuracy while preserving interpretability.

### Highlights
- **Novelty**: A new, multimodal Prototype-Based Network (mPBN) that fuses images with generated captions, extending **ProtoPNet** [1] with a **VisualBERT** [2] encoder. Prototypes live in the joint semantic space of text–vision.
- **Interpretability**: Classify-by-prototypes with explicit cluster and separation terms; additional **DeepSHAP** [3][4] analyses for both unimodal and multimodal models.
- **Results**: On Food-101, the best multimodal model (BLIP2 captions) achieves **77.34%** accuracy, surpassing the best unimodal ProtoPNet (**69.74%**) and VGG16 baseline (**55.37%**).
- **Engineering**: End-to-end pipeline: caption generation (BLIP/BLIP2), data curation, efficient dataloading, multimodal fusion, 3-stage ProtoPNet training, evaluation, and explainability artifacts.

<p style="text-align:center"><b>Original PBN Architecture</p>
<figure>
  <img width="800" height="600" alt="PBN Architecture" 
       src="https://github.com/user-attachments/assets/ab64c2f4-9ce5-4534-b4a6-96faabc4c643"
       style="display:block; margin:auto;" />
</figure>

<br><br><br>

<p style="text-align:center"><b>Novel mPBN Architecture</p>
<figure>
  <img width="800" height="600" alt="mPBN Architecture" 
       src="https://github.com/user-attachments/assets/f57c0dd4-8ccc-40a3-8f79-4b7866dcfb04"
       style="display:block; margin:auto;" />
</figure>


---
# Project Description

## 1) Overview
- **Objective**: Build an interpretable, high-accuracy food classifier that leverages both image content and natural-language descriptions, enabling prototype-level reasoning grounded in multimodal semantics.
- **Key idea**: Generate captions for each image (BLIP/BLIP2) [5][6], tokenize them, and feed the text together with CNN features into **VisualBERT**. Learn class-specific prototypes in the VisualBERT [CLS] space and classify by proximity to these prototypes.
- **Findings**:
  - Using captions substantially improves Food-101 accuracy vs image-only baselines.
  - Caption quality matters: BLIP2 captions outperform BLIP.
  - A small number of prototypes per class (1–2) works best in the multimodal setting; too many prototypes can hurt generalization.

For more context, see the final thesis submission `MSc_AI_Thesis_yyg760.pdf`.

---

## 2) Methodology

### Data & Captions
- **Dataset**: Food-101 (101 classes) [7]. Custom annotations and label mappings are used.
  - Dataloader: `Food101_dataloader.py` with `Food101Dataset` and helpers `lazy_load_original` / `lazy_load_customized`.
  - Inputs: images normalized to VGG/ResNet stats; labels mapped via `Food101/food-101/meta/label_map.json`.
- **Captioning**: `Food101_captioning.py` with pretrained Hugging Face models (BLIP and BLIP-2)
  - Captions are cleaned (noise removal + label leakage prevention) and saved into JSON annotations in `complete_annotations/`.

### Models 

#### Multimodal PBN (mPBN)
- Core: `multimodal_PBN.py`
  - Visual features: `VGG16` or "ResNet34" backbone features projected to match VisualBERT’s expected dimension.
  - Text features: tokenized captions (`bert-base-uncased`) into **VisualBERT** (`uclanlp/visualbert-vqa-coco-pre`).
  - Fusion: Visual token (from VGG16) + text tokens in VisualBERT → use `[CLS]` embedding as joint representation.
  - Prototypes: learn class-specific prototypes in the `[CLS]` space; logits are negative distances to prototypes via a linear last layer.

#### Baselines & Comparisons
- Image-only baselines: `vgg16_model.py`, `resnet34.py`.
- Unimodal ProtoPNet (image-only): `unimodal_ProtoPNet.py` (also three-stage training, spatial prototypes).

#### Training flow for baseline PBN and mPBN: 
- `ProtoPNet/train_and_test_mPBN.py` with the canonical ProtoPNet three stages
  1. Warm-up (train add-on/prototypes/last layer)
  2. Joint training (unfreeze all: encoder, CNN features, projection, prototypes, last layer)
  3. Last-layer optimization (optional convex-like step, L1 on last layer)
- Losses: Cross-entropy + prototype cluster/separation terms (+ optional L1), following ProtoPNet design (`ProtoPNet/settings.py`).
- Current project trained with 1P, 2P, 5P and 10P (_x_P = _x_ prototypes per class, see `MSc_AI_Thesis_yyg760.pdf` for more detail).

### Explainability: DeepSHAP for both unimodal and multimodal
  - Unimodal: `DeepSHAP_explaner_PBN*.py`.
  - Multimodal: `DeepSHAP_explaner_mPBN*.py`,  (wrappers prepare visual + repeated text inputs).
---

## 3) Technical Configuration

### Main Libraries & Dependencies
- Python: **3.12.7**
- Core libraries: `torch` (12.6, CUDA), `torchvision`, `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `scikit-image`
- Base PBN architecture (functioning itself as an unimodal PBN with a CNN backbone encoder): [ProtoPNet](https://github.com/cfchen-duke/ProtoPNet)  
- Hugging Face Transformers (pretrained):
   - Multimodal backbone encoder for base PBN (not by default, needed to implement manually): [VisualBERT](https://github.com/uclanlp/visualbert)
   - Caption generation: [BLIP](https://huggingface.co/docs/transformers/en/model_doc/blip), [BLIP2](https://huggingface.co/docs/transformers/en/model_doc/blip-2))
- Attention Visualization: [DeepSHAP](https://shap.readthedocs.io/en/latest/generated/shap.DeepExplainer.html), to interpret and visualize how both image (pixels) and text (tokens) contribute to model predictions in unimodal and multimodal settings.

See full list in `requirements.txt`.

### External Data & Resources
- **Dataset:** [Food-101](https://www.kaggle.com/datasets/dansbecker/food-101) (must be downloaded separately)
- Generated captions: Generated with _BLIP_ and _BLIP-2_ implemented in `Food101_captioning.py`, saved in `complete_annotations/*` in JSON format (original scripts to batch generate captions for the entire dataset are not included in this repository - please contact the author for the source).

### Model Training
- Multimodal training entry points:
  - Programmatic: call `train_multimodal_PBN(...)` from `multimodal_PBN.py`.
  - The training stages and optimizers are orchestrated by `ProtoPNet/train_and_test_mPBN.py` with hyperparameters in `ProtoPNet/settings.py`.
- Unimodal ProtoPNet: `train_protopnet(...)` in `unimodal_ProtoPNet.py`.
- Baselines: `train_vgg16(...)`,  `train_resnet34(...)`.
- **All training results will return a .pth file**

### Model Evaluation
- Each module provides `test_*` functions and **checkpoints (.pth file)**; aggregated CSV/JSON are written into `results_final/`(not published, please contact the author for complete result).

### Attention Visualization (DeepSHAP)
- **Core Scripts:**
  - `DeepSHAP_explaner_PBN*.py` — For unimodal (image-only) models.
  - `DeepSHAP_explaner_mPBN*.py` — For multimodal (image+caption) models, handling joint visual and text features.
- **Process Overview:**
  - Loads trained model checkpoints and test data.
  - Computes SHAP values for each prediction, highlighting influential pixels and tokens.
  - Generates visual artifacts (heatmaps, token importances) for qualitative analysis.
- **Output Artifacts:**
  - Visual explanations (e.g., saliency maps overlayed on images).
  - Token-level importance visualizations for captions.
  - Classification reports and confusion matrices (saved in `results_final/`).

**All trainings, evaluations, and attention analysis were completed on DAS-6 Supercomputer [8].**

  
### Repository Structure (key scripts and folders)
- `multimodal_PBN.py` — Multimodal model core
- `unimodal_ProtoPNet.py`, `vgg16_model.py`, `resnet34.py` — Baseline and unimodal models
- `ProtoPNet/` — ProtoPNet core logic, training, and settings
- `Food101_dataloader.py` — Data loaders
- `Food101_captioning.py` — Caption generation with BLIP/BLIP2
- `DeepSHAP_explaner_*` — SHAP explainers for model interpretability
- `model_training_scripts/*`, `SHAP_scripts/*` — Training and SHAP analysis scripts
- `results_final/` — Output results and reports
- `MSc_AI_Thesis_yyg760.pdf` - Final thesis submission

---

## 4) Results & Conclusion

### Key Results (Food-101)

| Model                            | Accuracy (%) | Notes                  |
|----------------------------------|--------------|------------------------|
| Baseline VGG16                   | 55.37        | Image-only             |
| PBN-ResNet34 (2 prototypes)      | 69.74        | Best unimodal PBN      |
| mPBN (BLIP captions, 1 prototype)| 72.65        | Multimodal (BLIP)      |
| mPBN (BLIP2 captions, 1 prototype)| 77.34       | Best overall (BLIP2)   |

(To access the complete result, please check `MSc_AI_Thesis_yyg760.pdf` or contact the author)

### Findings
- **Multimodality helps**: Adding captions lifts accuracy by ~7–8 points over the best unimodal ProtoPNet.
- **Caption quality matters**: BLIP2 > BLIP, consistent with stronger language modeling.
- **Prototype count**: 1–2 per class works best for mPBN. Larger counts can overfit and degrade performance.
- **Interpretability**: Prototypes in the joint space remain human-inspectable; SHAP analyses further explain visual/text contributions per prediction.

### Conclusion
- This project demonstrates that aligning prototypes with multimodal semantics produces a more accurate and still interpretable classifier. Language grounding via high-quality generated captions is a practical and scalable path to boost performance without human annotations. However, the performance of PBN models with multimodality can vary across several factors, such as the quality of the caption, the number of prototypes, model complexity, etc. Future study is required to discover a "gonden standard" to improve robustness of such multimodality task.

---

# Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/BruceLeo99/Msc-AI-thesis.git
   cd Msc-AI-thesis
- Use Python 3.12.7
  
2. Install dependencies
   ```bash
   pip install -r requirements.txt

3. Download and prepare Food-101 dataset and save under `Food101/food-101/meta/` (See download link above)

4. (Optional) Generate image captions by using `Food101_captioning.py` (Captions have already been generated and saved under `complete_annotations/*`. For the original batch generating script, don't hesitate to get in touch with the author)

5. Model Training: Use provided scripts in `model_training_scripts/` or run programmatically, e.g.:
  ```python
from Food101_dataloader import lazy_load_original
from multimodal_PBN import train_multimodal_PBN

train, test = lazy_load_original(load_captions=True, transform='vgg16', target_transform='integer', caption_type='blip2')
model_path = train_multimodal_PBN(train, test, model_name='mPBN_1p_blip2', device='cuda')
 ```

6. Evaluate and visualize results
- Scripts in `SHAP_scripts/*` for explainability.
- Results are saved in `results_final/`.
- Multimodal SHAP analysis tools are implemented in `DeepSHAP_explaner_mPBN*.py`
- Unimodal SHAP analysis tools are implemented in `DeepSHAP_explaner_PBN*.py`
  
Hardware: a CUDA-enabled GPU is strongly recommended for BLIP2/VisualBERT and training.

---


## References & Acknowledgements
<ol>
<li>Chen, C., Li, O., Tao, A., Barnett, A., Su, J., Rudin, C.: This looks like that:
 Deep learning for interpretable image recognition. arXiv preprint arXiv:1806.10574
 (2018), https://arxiv.org/abs/1806.10574</li>
<li>Li, L. H., Yatskar, M., Yin, D., Hsieh, C., & Chang, K. (2019). VisualBERT: A Simple and Performant Baseline for Vision and Language. ArXiv. https://arxiv.org/abs/1908.03557</li>
<li> Liu, M., Ning, Y., Yuan, H., Ong, M.E.H., Liu, N.: Balanced background and expla
nation data are needed in explaining deep learning models with shap: An empirical
 study on clinical decision making (2022), https://arxiv.org/abs/2206.04050</li>
<li> Lundberg, S., Lee, S.I.: A unified approach to interpreting model predictions
(2017), https://arxiv.org/abs/1705.07874</li>
<li>Li, J., Li, D., Xiong, C., Hoi, S.: Blip: Bootstrapping language-image pre
training for unified vision-language understanding and generation. In: Proceed
ings of the 39th International Conference on  Machine Learning (ICML) (2022),
 https://arxiv.org/abs/2201.12086</li>
<li> Li, J., Li, D., Savarese, S., Hoi, S.: BLIP-2: Bootstrapping language-image pre
training with frozen image encoders and large language models. In: Krause, A.,
 Brunskill, E., Cho, K., Engelhardt, B., Sabato, S., Scarlett, J. (eds.) Proceed
ings of the 40th International Conference on Machine Learning. Proceedings of
 Machine Learning Research, vol. 202, pp. 19730–19742. PMLR (23–29 Jul 2023),
 https://proceedings.mlr.press/v202/li23q.html</li>
<li>Bossard, L., Guillaumin, M., Gool, L.V.: Food-101– mining discriminative compo
nents with random forests. In: Proceedings of the European Conference on Com
puter Vision (ECCV). pp. 446–461. Springer, Zurich, Switzerland (2014)</li>
<li>Henri Bal, Dick Epema, Cees de Laat, Rob van Nieuwpoort, John Romein, Frank
 Seinstra, Cees Snoek, and Harry Wijshoff: "A Medium-Scale Distributed System for
 Computer Science Research: Infrastructure for the Long Term", IEEE Computer,
 Vol. 49, No. 5, pp. 54-63, May 2016.
</li>
</ol>

For research and implementation details, see the thesis PDF: `MSc_AI_Thesis_yyg760.pdf`.

---

## Notes for Reviewers (Engineering Readiness)
- Clear separation of concerns: data prep, captioning, dataloading, training, evaluation, and explainability.
- ProtoPNet patterns preserved for reproducibility; multimodal extension is modular and configurable.
- Results are scripted with CSV/JSON artifacts for easy comparison and auditability. 
