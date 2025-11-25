import json
import os
from datetime import datetime
from datasets import load_dataset

class EvaluateAugmentedDataset:
    """
    Compares the original dataset with the stitched augmented dataset.
    Computes accuracy for 'liked', 'seen', and 'suggested' fields.
    """

    def __init__(self, original_dataset_path=None, stitched_path="stitched_conversations.json", save_path=None):
        """
        Args:
            original_dataset_path (str or None): Path to the filtered original dataset. 
                If None, attempts to load 'test_set.json' first, then falls back to full ReDial.
            stitched_path (str): Path to the stitched JSON file.
            save_path (str or None): Path to save the evaluation results. 
                Defaults to '<stitched_dir>/evaluation_results.json'.
        """
        # Load stitched dataset first
        with open(stitched_path, "r", encoding="utf-8") as f:
            self.stitched_dataset = json.load(f)
        
        # Load original dataset with smart defaults
        if original_dataset_path is None:
            # Try to load the filtered test_set.json first
            test_set_path = "test_set.json"
            if os.path.exists(test_set_path):
                print(f"Loading filtered test set from '{test_set_path}'...")
                with open(test_set_path, "r", encoding="utf-8") as f:
                    self.original_dataset = json.load(f)
            else:
                print("WARNING: 'test_set.json' not found. Loading full ReDial test split from HuggingFace...")
                print("This may cause evaluation mismatches if conversations were filtered during processing.")
                self.original_dataset = load_dataset("community-datasets/re_dial")["test"].to_list()
        else:
            print(f"Loading original dataset from '{original_dataset_path}'...")
            with open(original_dataset_path, "r", encoding="utf-8") as f:
                self.original_dataset = json.load(f)

        self.save_path = save_path or os.path.join(
            os.path.dirname(os.path.abspath(stitched_path)),
            "evaluation_results.json"
        )

        print(f"Loaded {len(self.original_dataset)} original and {len(self.stitched_dataset)} stitched conversations.")
        
        # Verify dataset alignment
        if len(self.original_dataset) != len(self.stitched_dataset):
            print(f"\nWARNING: Dataset size mismatch!")
            print(f"  - Original: {len(self.original_dataset)} conversations")
            print(f"  - Stitched: {len(self.stitched_dataset)} conversations")
            print(f"  - Will only evaluate the first {min(len(self.original_dataset), len(self.stitched_dataset))} conversations")
            print(f"  - This may indicate that conversations were filtered during augmentation.\n")
        
        print(f"Results will be saved to: {self.save_path}")

    def _extract_answers(self, conversation):
        """Return the active list of answers (initiator or respondent)."""
        if conversation.get("initiatorQuestions"):
            return conversation["initiatorQuestions"]
        return conversation.get("respondentQuestions", [])

    def _compare_answers(self, original_answers, stitched_answers):
        """
        Compare answers movie-by-movie for liked/seen/suggested accuracy.
        Also tracks true positives, false positives, and false negatives for precision/recall.
        """
        # Accuracy tracking
        correct_liked, total_liked = 0, 0
        correct_seen, total_seen = 0, 0
        correct_suggested, total_suggested = 0, 0
        
        # Precision/Recall tracking
        tp_liked, fp_liked, fn_liked = 0, 0, 0
        tp_seen, fp_seen, fn_seen = 0, 0, 0
        tp_suggested, fp_suggested, fn_suggested = 0, 0, 0

        stitched_lookup = {a["movieId"]: a for a in stitched_answers if "movieId" in a}
        orig_lookup = {a["movieId"]: a for a in original_answers if "movieId" in a}

        # Process all movies in original (for recall - finding ground truth)
        for orig in original_answers:
            movie_id = orig.get("movieId")
            stitched = stitched_lookup.get(movie_id, {})

            # Compare liked
            if "liked" in orig and orig["liked"] != 2:
                total_liked += 1
                if stitched.get("liked") == orig["liked"]:
                    correct_liked += 1
                    tp_liked += 1
                else:
                    fn_liked += 1  # Ground truth exists but not predicted correctly

            # Compare seen
            if "seen" in orig and orig["seen"] != 2:
                total_seen += 1
                if stitched.get("seen") == orig["seen"]:
                    correct_seen += 1
                    tp_seen += 1
                else:
                    fn_seen += 1

            # Compare suggested
            if "suggested" in orig and orig["suggested"] != 2:
                total_suggested += 1
                if stitched.get("suggested") == orig["suggested"]:
                    correct_suggested += 1
                    tp_suggested += 1
                else:
                    fn_suggested += 1
        
        # Process movies only in stitched (for precision - false positives)
        for stitched in stitched_answers:
            movie_id = stitched.get("movieId")
            if movie_id not in orig_lookup:
                # Movie predicted but not in ground truth
                if "liked" in stitched and stitched["liked"] != 2:
                    fp_liked += 1
                if "seen" in stitched and stitched["seen"] != 2:
                    fp_seen += 1
                if "suggested" in stitched and stitched["suggested"] != 2:
                    fp_suggested += 1

        return {
            "liked": (correct_liked, total_liked),
            "seen": (correct_seen, total_seen),
            "suggested": (correct_suggested, total_suggested),
            "tp_liked": tp_liked,
            "fp_liked": fp_liked,
            "fn_liked": fn_liked,
            "tp_seen": tp_seen,
            "fp_seen": fp_seen,
            "fn_seen": fn_seen,
            "tp_suggested": tp_suggested,
            "fp_suggested": fp_suggested,
            "fn_suggested": fn_suggested,
        }

    def evaluate(self):
        """
        Evaluate stitched dataset against original dataset.
        Computes accuracy, precision, and recall.
        Saves and returns dict with all metrics.
        """
        liked_correct = seen_correct = suggested_correct = 0
        liked_total = seen_total = suggested_total = 0
        
        # Precision/Recall counters
        tp_liked_total = fp_liked_total = fn_liked_total = 0
        tp_seen_total = fp_seen_total = fn_seen_total = 0
        tp_suggested_total = fp_suggested_total = fn_suggested_total = 0
        
        # Track per-conversation results for debugging
        mismatches = []
        
        # Only evaluate up to the minimum length to handle size mismatches
        num_to_evaluate = min(len(self.original_dataset), len(self.stitched_dataset))

        for idx in range(num_to_evaluate):
            orig_conv = self.original_dataset[idx]
            stitched_conv = self.stitched_dataset[idx]
            
            # Optional: Verify conversation IDs match if available
            if "_original_conversation_id" in stitched_conv and "conversationId" in orig_conv:
                if stitched_conv["_original_conversation_id"] != orig_conv["conversationId"]:
                    print(f"WARNING: Conversation ID mismatch at index {idx}!")
                    print(f"  - Original ID: {orig_conv['conversationId']}")
                    print(f"  - Stitched ID: {stitched_conv['_original_conversation_id']}")
                    mismatches.append(idx)
            
            stitched_answers = self._extract_answers(stitched_conv)
            original_answers = self._extract_answers(orig_conv)

            result = self._compare_answers(original_answers, stitched_answers)
            
            # Accuracy counts
            liked_correct += result["liked"][0]
            liked_total += result["liked"][1]
            seen_correct += result["seen"][0]
            seen_total += result["seen"][1]
            suggested_correct += result["suggested"][0]
            suggested_total += result["suggested"][1]
            
            # Precision/Recall counts
            tp_liked_total += result["tp_liked"]
            fp_liked_total += result["fp_liked"]
            fn_liked_total += result["fn_liked"]
            tp_seen_total += result["tp_seen"]
            fp_seen_total += result["fp_seen"]
            fn_seen_total += result["fn_seen"]
            tp_suggested_total += result["tp_suggested"]
            fp_suggested_total += result["fp_suggested"]
            fn_suggested_total += result["fn_suggested"]

        def safe_div(a, b):
            return round(a / b, 3) if b > 0 else 0.0
        
        # Calculate precision and recall for each field
        liked_precision = safe_div(tp_liked_total, tp_liked_total + fp_liked_total)
        liked_recall = safe_div(tp_liked_total, tp_liked_total + fn_liked_total)
        liked_f1 = safe_div(2 * liked_precision * liked_recall, liked_precision + liked_recall)
        
        seen_precision = safe_div(tp_seen_total, tp_seen_total + fp_seen_total)
        seen_recall = safe_div(tp_seen_total, tp_seen_total + fn_seen_total)
        seen_f1 = safe_div(2 * seen_precision * seen_recall, seen_precision + seen_recall)
        
        suggested_precision = safe_div(tp_suggested_total, tp_suggested_total + fp_suggested_total)
        suggested_recall = safe_div(tp_suggested_total, tp_suggested_total + fn_suggested_total)
        suggested_f1 = safe_div(2 * suggested_precision * suggested_recall, suggested_precision + suggested_recall)
        
        # Overall metrics
        overall_tp = tp_liked_total + tp_seen_total + tp_suggested_total
        overall_fp = fp_liked_total + fp_seen_total + fp_suggested_total
        overall_fn = fn_liked_total + fn_seen_total + fn_suggested_total
        
        overall_precision = safe_div(overall_tp, overall_tp + overall_fp)
        overall_recall = safe_div(overall_tp, overall_tp + overall_fn)
        overall_f1 = safe_div(2 * overall_precision * overall_recall, overall_precision + overall_recall)

        metrics = {
            # Accuracy metrics
            "liked_accuracy": safe_div(liked_correct, liked_total),
            "liked_correct": liked_correct,
            "liked_total": liked_total,
            "seen_accuracy": safe_div(seen_correct, seen_total),
            "seen_correct": seen_correct,
            "seen_total": seen_total,
            "suggested_accuracy": safe_div(suggested_correct, suggested_total),
            "suggested_correct": suggested_correct,
            "suggested_total": suggested_total,
            "overall_accuracy": safe_div(
                liked_correct + seen_correct + suggested_correct,
                liked_total + seen_total + suggested_total
            ),
            "overall_correct": liked_correct + seen_correct + suggested_correct,
            "overall_total": liked_total + seen_total + suggested_total,
            
            # Precision/Recall metrics for liked
            "liked_precision": liked_precision,
            "liked_recall": liked_recall,
            "liked_f1": liked_f1,
            "liked_tp": tp_liked_total,
            "liked_fp": fp_liked_total,
            "liked_fn": fn_liked_total,
            
            # Precision/Recall metrics for seen
            "seen_precision": seen_precision,
            "seen_recall": seen_recall,
            "seen_f1": seen_f1,
            "seen_tp": tp_seen_total,
            "seen_fp": fp_seen_total,
            "seen_fn": fn_seen_total,
            
            # Precision/Recall metrics for suggested
            "suggested_precision": suggested_precision,
            "suggested_recall": suggested_recall,
            "suggested_f1": suggested_f1,
            "suggested_tp": tp_suggested_total,
            "suggested_fp": fp_suggested_total,
            "suggested_fn": fn_suggested_total,
            
            # Overall precision/recall
            "overall_precision": overall_precision,
            "overall_recall": overall_recall,
            "overall_f1": overall_f1,
            "overall_tp": overall_tp,
            "overall_fp": overall_fp,
            "overall_fn": overall_fn,
            
            # Meta information
            "conversations_evaluated": num_to_evaluate,
            "conversations_with_id_mismatch": len(mismatches),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        print("\n" + "="*80)
        print("EVALUATION RESULTS".center(80))
        print("="*80)
        
        print(f"\nConversations Evaluated: {metrics['conversations_evaluated']}")
        if mismatches:
            print(f"⚠️  Conversations with ID mismatches: {len(mismatches)}")
            print(f"   Indices: {mismatches[:10]}" + (" ..." if len(mismatches) > 10 else ""))
        
        print(f"\n{'Metric':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Correct/Total'}")
        print("-" * 80)
        print(f"{'Liked':<15} {metrics['liked_accuracy']:.3f}      {metrics['liked_precision']:.3f}      {metrics['liked_recall']:.3f}      {metrics['liked_f1']:.3f}      {metrics['liked_correct']}/{metrics['liked_total']}")
        print(f"{'Seen':<15} {metrics['seen_accuracy']:.3f}      {metrics['seen_precision']:.3f}      {metrics['seen_recall']:.3f}      {metrics['seen_f1']:.3f}      {metrics['seen_correct']}/{metrics['seen_total']}")
        print(f"{'Suggested':<15} {metrics['suggested_accuracy']:.3f}      {metrics['suggested_precision']:.3f}      {metrics['suggested_recall']:.3f}      {metrics['suggested_f1']:.3f}      {metrics['suggested_correct']}/{metrics['suggested_total']}")
        print("-" * 80)
        print(f"{'Overall':<15} {metrics['overall_accuracy']:.3f}      {metrics['overall_precision']:.3f}      {metrics['overall_recall']:.3f}      {metrics['overall_f1']:.3f}      {metrics['overall_correct']}/{metrics['overall_total']}")
        
        print("\n" + "Detailed Counts".center(80))
        print("-" * 80)
        print(f"{'Metric':<15} {'True Positives':<18} {'False Positives':<18} {'False Negatives':<18}")
        print("-" * 80)
        print(f"{'Liked':<15} {metrics['liked_tp']:<18} {metrics['liked_fp']:<18} {metrics['liked_fn']:<18}")
        print(f"{'Seen':<15} {metrics['seen_tp']:<18} {metrics['seen_fp']:<18} {metrics['seen_fn']:<18}")
        print(f"{'Suggested':<15} {metrics['suggested_tp']:<18} {metrics['suggested_fp']:<18} {metrics['suggested_fn']:<18}")
        print("-" * 80)
        print(f"{'Overall':<15} {metrics['overall_tp']:<18} {metrics['overall_fp']:<18} {metrics['overall_fn']:<18}")
        print("="*80 + "\n")

        # Save metrics to JSON file
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        print(f"✅ Saved evaluation results to '{self.save_path}'")

        return metrics


if __name__ == "__main__":
    evaluator = EvaluateAugmentedDataset(
        stitched_path="stitched_conversations.json"
    )
    metrics = evaluator.evaluate()