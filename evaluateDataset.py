import argparse
import json
import os
import math
from datetime import datetime
from datasets import load_dataset
from tqdm import tqdm

class EvaluateAugmentedDataset:
    """
    Compares the original dataset with the stitched augmented dataset.
    Computes Accuracy, Precision, Recall, and F1-Score for 'liked', 'seen', and 'suggested' fields.
    """

    def __init__(self, original_dataset_path=None, stitched_path="augmented_full_dataset.json", save_path=None, args=None):
        """
        Args:
            original_dataset_path (str or None): Path to the filtered original dataset (e.g., 'test_set.json').
            stitched_path (str): Path to the augmented JSON file created by test.py.
            save_path (str or None): Path to save the evaluation results.
            args (argparse.Namespace): Parsed CLI args including --original_snippets_dir
        """
        self.args = args

        # --- Load Stitched Dataset (The Model's Output) ---
        if not os.path.exists(stitched_path):
            raise FileNotFoundError(f"Stitched file not found: {stitched_path}")

        with open(stitched_path, "r", encoding="utf-8") as f:
            self.stitched_dataset = json.load(f)

        # --- Load Original Dataset (Ground Truth) ---
        self.original_dataset = []

        # Try loading from original_snippets_dir if flag present
        if self.args.original_snippets_dir:
            orig_dir = self.args.original_snippets_dir
            print(f"Loading original conversations for ground truth from directory: {orig_dir}")

            if not os.path.exists(orig_dir):
                raise FileNotFoundError(f"Original snippets directory not found: {orig_dir}")

            files = sorted([
                f for f in os.listdir(orig_dir)
                if os.path.isfile(os.path.join(orig_dir, f))
            ])

            for file in files:
                path = os.path.join(orig_dir, file)
                try:
                    # Detect if JSON conversation or raw text
                    with open(path, "r", encoding="utf-8") as fp:
                        content = fp.read().strip()
                        try:
                            # attempt to parse JSON
                            conv = json.loads(content)
                            self.original_dataset.append(conv)
                        except json.JSONDecodeError:
                            # fallback to raw string
                            self.original_dataset.append({"raw": content})
                except Exception as e:
                    print(f"ERROR loading {file}: {e}")

            print(f"Loaded {len(self.original_dataset)} original conversations from directory.")

            # Align size with augmented
            num_snippets = len(self.original_dataset)
            num_aug = len(self.stitched_dataset)

            num_shots_skipped = num_snippets - num_aug
            if 0 < num_shots_skipped < num_snippets:
                print(f"Skipping the first {num_shots_skipped} conversations in original data to match augmented (generator skipped shots).")
                self.original_dataset = self.original_dataset[num_shots_skipped:]
            elif num_shots_skipped <= 0:
                print("No conversations were skipped or sizes identical.")

            if len(self.original_dataset) > num_aug:
                self.original_dataset = self.original_dataset[:num_aug]

        #  Else try test_set.json
        elif original_dataset_path:
            print(f"Loading original dataset from '{original_dataset_path}'...")
            with open(original_dataset_path, "r", encoding="utf-8") as f:
                self.original_dataset = json.load(f)

        elif os.path.exists("test_set.json"):
            print("Loading original conversations from 'test_set.json'...")
            with open("test_set.json", "r", encoding="utf-8") as f:
                self.original_dataset = json.load(f)

        # Else fallback to full HF load
        else:
            print("WARNING: No comparison file found, loading FULL ReDial dataset from HuggingFace for ground truth...")
            redial = load_dataset("community-datasets/re_dial")
            for split in ["train", "test", "validation"]:
                if split in redial:
                    self.original_dataset.extend(redial[split].to_list())

            # Align size
            num_shots_skipped = len(self.original_dataset) - len(self.stitched_dataset)
            if 0 < num_shots_skipped < len(self.original_dataset):
                print(f"Skipping first {num_shots_skipped} conversations to match augmented size.")
                self.original_dataset = self.original_dataset[num_shots_skipped:]
            else:
                print("No conversations were skipped or sizes identical.")

        # --- Save path naming logic ---
        base_json_name = os.path.basename(os.path.abspath(stitched_path))
        self.save_path = save_path or os.path.join(
            os.path.dirname(os.path.abspath(stitched_path)),
            "evaluation_results_" + os.path.splitext(base_json_name)[0] + ".json"
        )

        print(f"Loaded {len(self.original_dataset)} original and {len(self.stitched_dataset)} stitched conversations.")
        print(f"Results will be saved to: {self.save_path}")

    def _extract_answers(self, conversation):
        # If conversation is a list, search inside it (fallback)
        if isinstance(conversation, list):
            for item in conversation:
                if isinstance(item, dict) and "initiatorQuestions" in item:
                    return item["initiatorQuestions"]
                if isinstance(item, dict) and "respondentQuestions" in item:
                    return item["respondentQuestions"]
            return []  # nothing found

        # Normal case: conversation is a dict
        if isinstance(conversation, dict):
            if "initiatorQuestions" in conversation:
                return conversation["initiatorQuestions"]
            if "respondentQuestions" in conversation:
                return conversation["respondentQuestions"]
            if "movieId" in conversation:
                return [conversation]  # allow single-answer fallback

        return []


    def _compare_answers(self, original_answers, stitched_answers):
        """
        Compare annotations for a single conversation. Now INCLUDES label = 2.
        """

        if not isinstance(original_answers, list):
            return {"liked":{"tp":0,"fp":0,"fn":0,"correct":0,"total":0},
                    "seen":{"tp":0,"fp":0,"fn":0,"correct":0,"total":0},
                    "suggested":{"tp":0,"fp":0,"fn":0,"correct":0,"total":0}}

        if not isinstance(stitched_answers, list):
            stitched_answers = []

        # --- Build Ground Truth map (INCLUDES 2) ---
        orig_map = {}
        stitched_map = {}

        for a in original_answers:
            if isinstance(a, dict) and "movieId" in a:
                orig_map[a["movieId"]] = {
                    "liked": a.get("liked"),
                    "seen": a.get("seen"),
                    "suggested": a.get("suggested")
                }

        for a in stitched_answers:
            if isinstance(a, dict) and "movieId" in a:
                stitched_map[a["movieId"]] = {
                    "liked": a.get("liked"),
                    "seen": a.get("seen"),
                    "suggested": a.get("suggested")
                }

        # --- Initialize metric counters ---
        res = {
            "liked": {"tp": 0, "fp": 0, "fn": 0, "correct": 0, "total": 0},
            "seen": {"tp": 0, "fp": 0, "fn": 0, "correct": 0, "total": 0},
            "suggested": {"tp": 0, "fp": 0, "fn": 0, "correct": 0, "total": 0},
        }
        all_ids = set(orig_map.keys()) | set(stitched_map.keys())

        for mid in all_ids:
            orig = orig_map.get(mid, {})
            pred = stitched_map.get(mid, {})

            for f in ["liked", "seen", "suggested"]:
                o = orig.get(f)
                p = pred.get(f)

                # normalize None to  2 (unknown)
                o = 2 if o is None else o
                p = 2 if p is None else p

                # Only evaluate GT values 0 and 1 for recall/accuracy
                if o < 2:  
                    res[f]["total"] += 1

                    if p < 2:
                        if p == o:
                            res[f]["tp"] += 1
                            res[f]["correct"] += 1
                        else:
                            res[f]["fn"] += 1
                            res[f]["fp"] += 1
                    else:
                        # Model predicted 2 when GT was 0/1 → that's a miss (FN)
                        res[f]["fn"] += 1

                # Count FP when model predicts 0/1 but GT = 2
                elif o == 2 and p < 2:
                    res[f]["fp"] += 1

        return res


    def evaluate(self):
        global_metrics = {
            "liked": {"tp": 0, "fp": 0, "fn": 0, "correct": 0, "total": 0},
            "seen": {"tp": 0, "fp": 0, "fn": 0, "correct": 0, "total": 0},
            "suggested": {"tp": 0, "fp": 0, "fn": 0, "correct": 0, "total": 0},
        }

        n = min(len(self.original_dataset), len(self.stitched_dataset))
        
        for datapoint in tqdm(self.original_dataset, desc="Evaluating Results"):
            o = datapoint
            s = None
            r = None
            curr_conv = o[0].get("conversationId")
            
            for testpoint in (self.stitched_dataset):
                if testpoint.get("conversationId") == curr_conv:
                    s = testpoint
                    break
            if s is not None:
                r = self._compare_answers(self._extract_answers(o), self._extract_answers(s))
            
            for f in ["liked", "seen", "suggested"]:
                for k in global_metrics[f]:
                    global_metrics[f][k] += r[f][k]



        def sd(a, b): return round(a/b,4) if b else 0.0
        fm, otp, ofp, ofn, oc, ot = {}, 0,0,0,0,0
        for f in ["liked","seen","suggested"]:
            m = global_metrics[f]
            acc = sd(m["correct"], m["total"])
            pr = sd(m["tp"], m["tp"]+m["fp"])
            rc = sd(m["tp"], m["tp"]+m["fn"])
            f1 = sd(2*pr*rc, pr+rc) if (pr+rc) else 0.0
            fm.update({f"{f}_accuracy":acc, f"{f}_precision":pr, f"{f}_recall":rc, f"{f}_f1":f1,
                       f"{f}_tp":m["tp"], f"{f}_fp":m["fp"], f"{f}_fn":m["fn"]})
            otp+=m["tp"];ofp+=m["fp"];ofn+=m["fn"];oc+=m["correct"];ot+=m["total"]

        fm.update({"overall_accuracy": sd(oc,ot),
                   "overall_precision": sd(otp,otp+ofp),
                   "overall_recall": sd(otp,otp+ofn),
                   "overall_f1": sd(2*sd(otp,otp+ofp)*sd(otp,otp+ofn),
                                    sd(otp,otp+ofp)+sd(otp,otp+ofn)) if (otp+ofp+ofn) else 0,
                   "overall_tp":otp,"overall_fp":ofp,"overall_fn":ofn,
                   "conversations_evaluated":n,"timestamp":datetime.now().isoformat()})

        print(f"\nSaving evaluation results to: {self.save_path}")
        os.makedirs(os.path.dirname(self.save_path) or '.', exist_ok=True)
        with open(self.save_path,"w",encoding="utf-8") as f: json.dump(fm,f,indent=4)
        return fm

def print_results(metrics):
    print("\n" + "="*80)
    print("✨ AUGMENTED DATASET EVALUATION RESULTS ✨")
    print(f"Conversations Evaluated: {metrics['conversations_evaluated']}")
    print("="*80)
    print(f"{'Field':<10} | {'Acc':>6} | {'Prec':>6} | {'Rec':>6} | {'F1':>6} | {'TP':>6} | {'FP':>6} | {'FN':>6}")
    print("-"*80)
    for f in ["liked","seen","suggested"]:
        print(f"{f:<10} | {metrics[f'{f}_accuracy']:>6.4f} | {metrics[f'{f}_precision']:>6.4f} | "
              f"{metrics[f'{f}_recall']:>6.4f} | {metrics[f'{f}_f1']:>6.4f} | "
              f"{metrics[f'{f}_tp']:>6} | {metrics[f'{f}_fp']:>6} | {metrics[f'{f}_fn']:>6}")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_snippets_dir", type=str, default="runs/Qwen2.5-7B-Instruct_0shot/original_snippets",
                        help="Directory containing 1 file per original conversation in processing order.")
    parser.add_argument("--stitched_path", type=str,
                        default="runs/Qwen2.5-7B-Instruct_0shot/augmented_7B_0_shot.json",
                        help="Path to stitched augmented dataset JSON.")
    parser.add_argument("--save_path", type=str, default="runs/Qwen2.5-7B-Instruct_0shot/evaluation_results_augmented7B.json",
                        help="Optional custom path to save results.")

    args = parser.parse_args()

    evaluator = EvaluateAugmentedDataset(
        stitched_path=args.stitched_path,
        original_dataset_path=None,
        save_path=args.save_path,
        args=args
    )
    res = evaluator.evaluate()
    print_results(res)
