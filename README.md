# gsm_rand

### generate wording variations, where items (subjects) and numbers are set by seed
```
python ground_truth.py

# or

bash ground_truth.sh
```
generate graph and question, deductions in `out/seed42/question_variations.json`.
save total in `out/question_variations.json`.
This is target question.

### generate prompts and prepend to target question
```
python gen_prompts.py
```
save to `out/question_variations_with_context.json`.

### run results
```
python eval.py
```
save to `data/results`

### parse model result
generate descriptive analysis and tree like ground truth
```
python parse_result.py
```
