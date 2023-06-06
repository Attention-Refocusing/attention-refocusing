import json
import torch
import clip
from PIL import Image
import os
import numpy as np
import argparse
import pandas as pd

from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

from transformers import Blip2Processor, Blip2ForConditionalGeneration

def compute_max(scorer, gt_prompts, pred_prompts):
    scores = []
    for pred_prompt in pred_prompts:
        for gt_prompt in gt_prompts:
            cand = {0: [pred_prompt]}
            ref = {0: [gt_prompt]}
            score, _ = scorer.compute_score(ref, cand)
            scores.append(score)
        # import pdb; pdb.set_trace()
    return np.max(scores)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="model name")
    parser.add_argument("--file_path", type=str, help="prompt file")
    args = parser.parse_args()

    scorer_cider = Cider()
    bleu1 = Bleu(n=1)
    bleu2 = Bleu(n=2)
    bleu3 = Bleu(n=3)
    bleu4 = Bleu(n=4)
    tokenizer = PTBTokenizer()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    bit8 = False
    dtype = {'load_in_8bit': True} if bit8 else {'torch_dtype': torch.float16}
    model_name = 'Salesforce/blip2-opt-2.7b-coco'
    processor = Blip2Processor.from_pretrained(model_name)
    blip2 = Blip2ForConditionalGeneration.from_pretrained(model_name, **dtype)
    blip2.to(device)

    df = pd.read_csv(args.file_path)

    prompts = df['prompt']
    image_paths = df['imgs']
    aug_prompts = df['aug_prompt']

    cider_scores, bleu1_scores, bleu2_scores, bleu3_scores, bleu4_scores = [],[],[],[],[]
    for idx, text_prompts in enumerate(prompts):
        # import pdb; pdb.set_trace()
        text_prompts = [text_prompts]
        augs= aug_prompts[idx].split('|')
        text_prompts.extend(augs)
        # for i, image_path in enumerate(image_paths[idx]):
        image = Image.open(image_paths[idx]).convert('RGB')
        inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
        with torch.no_grad():
            generated_id = blip2.generate(**inputs, do_sample=True, top_p=0.95, num_return_sequences=5)
            generated_texts = processor.batch_decode(generated_id, skip_special_tokens=True)
            candidates = []
            for generated_text in generated_texts:
                candidates.append(generated_text.strip())
            cider_scores.append(compute_max(scorer_cider, text_prompts, candidates))
            bleu1_scores.append(compute_max(bleu1, text_prompts, candidates))
            bleu2_scores.append(compute_max(bleu2, text_prompts, candidates))
            bleu3_scores.append(compute_max(bleu3, text_prompts, candidates))
            bleu4_scores.append(compute_max(bleu4, text_prompts, candidates))

    print(np.mean(cider_scores), np.mean(bleu1_scores), np.mean(bleu2_scores), np.mean(bleu3_scores), np.mean(bleu4_scores))