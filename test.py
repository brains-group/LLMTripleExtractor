import argparse
import json
import math
import sys
import re
import random
from collections import defaultdict
import os
import torch
import statistics

from tqdm import tqdm
import datasets
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from augmentDataset import AugmentDataset
from evaluateDataset import EvaluateAugmentedDataset

random.seed(2025)
PROMPT = """Extract movie preferences from this conversation as triples.

**CRITICAL RULES:**
1. ONLY extract if the user EXPLICITLY states their preference
2. ONLY extract specific movie titles - NO genres, actors, or franchises
3. Subject must ALWAYS be "User" (not "You")
4. Use lowercase for relations: likes, dislikes, seen, notSeen, suggested
5. Output ONLY triples in the exact format shown below
6. If no triples can be extracted, output: "No triples found."

**SCHEMA:**
Format: (User, relation, Movie Title (Year))

Relations:
- likes: User explicitly says they like/love/enjoy a specific movie
- dislikes: User explicitly says they dislike/hate a specific movie
- seen: User explicitly states they watched a specific movie
- notSeen: User explicitly states they have NOT watched a specific movie
- suggested: Assistant recommends a specific movie to the user

{}

---

**NEW CONVERSATION TO EXTRACT:**
{}

**Extracted Triples:**"""

SHOT_TEMPLATE = """
### Example Task:

Conversation:
{}

Extracted Triples:
{}
"""

TRIPLE_FORMAT = "({}, {}, {})"

def messagesToConversation(datapoint):
    messages = datapoint["messages"]
    initiatorID = datapoint["initiatorWorkerId"]

    def replaceMovieMentions(message):
        for movieMention in datapoint["movieMentions"]:
            if movieMention["movieName"] is not None:
                message = message.replace(
                    "@" + movieMention["movieId"], 
                    movieMention["movieName"]
                )
        return message

    lines = []
    for message in messages:
        role = "User" if message["senderWorkerId"] == initiatorID else "Assistant"
        text = replaceMovieMentions(message["text"])
        lines.append(f"{role}: {text}")
    
    return "\n".join(lines)


def getTriples(datapoint):
    movieMentions = {
        movieMention["movieId"]: movieMention["movieName"]
        for movieMention in datapoint["movieMentions"]
    }
    answers = (
        datapoint["initiatorQuestions"]
        if len(datapoint["initiatorQuestions"]) > 0
        else datapoint["respondentQuestions"]
    )
    triples = []
    for answerSet in answers:
        if not (answerSet["liked"] == 2):
            triples.append(
                (
                    "User",
                    "likes" if answerSet["liked"] == 1 else "dislikes",
                    movieMentions[answerSet["movieId"]],
                )
            )
        if not (answerSet["seen"] == 2):
            triples.append(
                (
                    "User",
                    "seen" if answerSet["seen"] == 1 else "notSeen",
                    movieMentions[answerSet["movieId"]],
                )
            )
        if answerSet["suggested"] == 1:
            triples.append(
                (
                    "User",
                    "suggested",
                    movieMentions[answerSet["movieId"]],
                )
            )
    return triples


def triplesToString(triples):
    return [TRIPLE_FORMAT.format(*triple) for triple in triples]


def createShot(datapoint):
    return SHOT_TEMPLATE.format(
        messagesToConversation(datapoint),
        "\n".join(triplesToString(getTriples(datapoint))),
    )


def createShots(shots):
    if len(shots) <= 0:
        return ""
    return "\n".join([createShot(shot) for shot in shots])


# ============= PHASE 1: GENERATE RESPONSES =============
def generate_responses(dataset, shots, model, tokenizer, args, run_dir):
    """Phase 1: Generate model responses for all datapoints"""
    num_shots_str = str(len(shots)) + "shot"
    name = args.base_model_path.replace("/", "_") + "_" + num_shots_str
    
    responsesPath = os.path.join(run_dir, "responses", name + ".json")
    os.makedirs(os.path.dirname(responsesPath), exist_ok=True)
    
    if os.path.exists(responsesPath):
        print(f"Loading cached responses from {responsesPath}")
        with open(responsesPath, "r") as file:
            return json.load(file)
    
    print(f"\n{'='*60}")
    print(f"PHASE 1: Generating Model Responses")
    print(f"{'='*60}")
    
    responses = []
    batch_size = args.batch_size
    
    for batch_start in tqdm(range(0, len(dataset), batch_size), desc="Generating responses"):
        batch_end = min(batch_start + batch_size, len(dataset))
        batch_datapoints = dataset[batch_start:batch_end]
        batch_texts = []
        
        for dataPoint in batch_datapoints:
            text = tokenizer.apply_chat_template(
                [
                    {
                        "content": PROMPT.format(
                            createShots(shots), messagesToConversation(dataPoint)
                        ),
                        "role": "user",
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            batch_texts.append(text)
        
        # Generate responses for batch
        model_inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        ).to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens
        generated_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
        batch_responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        responses.extend(batch_responses)
    
    # Save responses
    print(f"Saving {len(responses)} responses to {responsesPath}")
    with open(responsesPath, "w") as file:
        json.dump(responses, file, indent=2)
    
    return responses


# ============= PHASE 2: AUGMENT DATASET =============
def augment_dataset(dataset, responses, args, run_dir):
    """Phase 2: Augment dataset using generated responses"""
    print(f"\n{'='*60}")
    print(f"PHASE 2: Augmenting Dataset")
    print(f"{'='*60}")
    
    os.makedirs(os.path.join(run_dir, "recommendations"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "original_snippets"), exist_ok=True)  
    
    augmented_conversations = []
    num_shots_str = str(args.num_shots) + "shot"
    
    for index, (dataPoint, response) in enumerate(tqdm(zip(dataset, responses), 
                                                       total=len(dataset), 
                                                       desc="Augmenting")):
        # Handle Qwen thinking tokens
        if "Qwen" in args.base_model_path:
            endThinkString = "</think>"
            endThinkIndex = response.rfind(endThinkString)
            if endThinkIndex != -1:
                response = response[(endThinkIndex + len(endThinkString)):]
        
        # Extract recommendations
        recommendations = re.findall(r"\([^,]+, [^,]+, [^\)]+\)?\)", response)
        
        # Save recommendations
        rec_path = os.path.join(run_dir, "recommendations", f"{num_shots_str}_{index}.json")
        with open(rec_path, "w", encoding="utf-8") as f:
            json.dump(recommendations, f, indent=2)
        
        # Save original snippet
        snippet_path = os.path.join(run_dir, "original_snippets", f"original_snippet_{num_shots_str}_{index}.json")
        with open(snippet_path, "w", encoding="utf-8") as f:
            json.dump([dataPoint], f, indent=2)
        
        # Perform augmentation
        try:
            augmenter = AugmentDataset()
            augmenter.load_conversation_from_data(dataPoint)
            augmenter.load_triples_from_list(recommendations)
            augmenter.update_conversation()
            augmented_conversations.append(augmenter.conversation)
        except Exception as e:
            print(f"Warning: Failed to augment conversation {index}: {e}")
            augmented_conversations.append(dataPoint)
    
    # Save augmented dataset
    output_path = os.path.join(run_dir, args.output_augmented_json)
    print(f"Saving augmented dataset to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(augmented_conversations, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved {len(augmented_conversations)} augmented conversations")
    return augmented_conversations


# ============= PHASE 3: EVALUATE =============
def print_results(metrics):
    """Prints the final metrics in a readable table format."""
    print("\n" + "="*80)
    print("✨ AUGMENTED DATASET EVALUATION RESULTS ✨")
    print(f"Conversations Evaluated: {metrics['conversations_evaluated']}")
    print("="*80)
    
    # Table Header
    print(f"{'Field':<10} | {'Acc':>6} | {'Prec':>6} | {'Rec':>6} | {'F1':>6} | {'TP':>6} | {'FP':>6} | {'FN':>6} | {'Total GT':>10}")
    print("-" * 80)
    
    fields = ["liked", "seen", "suggested"]
    for field in fields:
        print(f"{field:<10} | "
              f"{metrics[f'{field}_accuracy']:>6.4f} | "
              f"{metrics[f'{field}_precision']:>6.4f} | "
              f"{metrics[f'{field}_recall']:>6.4f} | "
              f"{metrics[f'{field}_f1']:>6.4f} | "
              f"{metrics[f'{field}_tp']:>6} | "
              f"{metrics[f'{field}_fp']:>6} | "
              f"{metrics[f'{field}_fn']:>6} | "
              f"{metrics[f'{field}_tp'] + metrics[f'{field}_fn']:>10}")
              
    print("-" * 80)
    
    # Overall Row
    print(f"{'OVERALL':<10} | "
          f"{metrics['overall_accuracy']:>6.4f} | "
          f"{metrics['overall_precision']:>6.4f} | "
          f"{metrics['overall_recall']:>6.4f} | "
          f"{metrics['overall_f1']:>6.4f} | "
          f"{metrics['overall_tp']:>6} | "
          f"{metrics['overall_fp']:>6} | "
          f"{metrics['overall_fn']:>6} | "
          f"{metrics['overall_tp'] + metrics['overall_fn']:>10}")
    print("="*80)


def evaluate_responses(dataset, responses, args):
    """Phase 3: Evaluate model responses"""
    print(f"\n{'='*60}")
    print(f"PHASE 3: Evaluating Responses")
    print(f"{'='*60}")
    
    truePositives = defaultdict(int)
    falsePositives = defaultdict(int)
    falseNegatives = defaultdict(int)
    hits = defaultdict(lambda: [0 for _ in range(10)])
    mrr = defaultdict(int)
    numDatapoints = 0
    
    for index, (dataPoint, response) in enumerate(tqdm(zip(dataset, responses), 
                                                       total=len(dataset), 
                                                       desc="Evaluating")):
        # Handle Qwen thinking tokens
        if "Qwen" in args.base_model_path:
            endThinkString = "</think>"
            endThinkIndex = response.rfind(endThinkString)
            if endThinkIndex != -1:
                response = response[(endThinkIndex + len(endThinkString)):]
        
        goals = getTriples(dataPoint)
        recommendations = re.findall(r"\([^,]+, [^,]+, [^\)]+\)?\)", response)
        
        formatFollowed = len(recommendations) > 0
        if formatFollowed:
            falsePositives[index] += len(recommendations)
        else:
            recommendations = [response]
        
        rank = -1
        doBreak = False
        for goal in goals:
            goalNotFound = True
            for recommendationIndex, recommendation in enumerate(recommendations):
                foundGoal = False
                recommendationSubstring = recommendation.lower()
                firstIndex = recommendationSubstring.find(goal[0].lower())
                if firstIndex >= 0:
                    recommendationSubstring = recommendationSubstring[
                        (firstIndex + len(goal[0])) :
                    ]
                    secondIndex = recommendationSubstring.find(goal[1].lower())
                    if secondIndex >= 0:
                        recommendationSubstring = recommendationSubstring[
                            (secondIndex + len(goal[1])) :
                        ]
                        movie_title = goal[2]
                        if " (" in movie_title:
                            movie_title = movie_title[: movie_title.find(" (")]
                        foundGoal = recommendationSubstring.find(movie_title.lower()) >= 0
                
                if foundGoal:
                    goalNotFound = False
                    truePositives[index] += 1
                    if formatFollowed:
                        falsePositives[index] -= 1
                        if recommendationIndex < rank or rank == -1:
                            rank = recommendationIndex
                    else:
                        rank = 0
                        falseNegatives[index] += len(goals) - (goals.index(goal) + 1)
                        doBreak = True
                    recommendations.remove(recommendation)
                    break
            
            if goalNotFound:
                falseNegatives[index] += 1
            if doBreak:
                break
        
        falseNegatives[index] = max(
            0, min(20 - truePositives[index], falseNegatives[index])
        )
        mrr[index] = 0
        if rank >= 0:
            mrr[index] += 1 / (rank + 1)
            if len(hits[index]) < len(recommendations):
                hits[index] += [hits[index][-1]] * (
                    len(recommendations) - len(hits[index])
                )
            for i in range(rank, len(hits[index]), 1):
                hits[index][i] += 1
        numDatapoints += 1
    
    def getSumOfDictVals(dictionary):
        return sum(dictionary.values())

    tp_sum = getSumOfDictVals(truePositives)
    fp_sum = getSumOfDictVals(falsePositives)
    fn_sum = getSumOfDictVals(falseNegatives)
    
    prec = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0
    rec = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0
    
    return (
        "\nOverall Stats\n"
        "Number of Tests: {num_tests}\n"
        "Precision: {precision}\n"
        "Recall: {recall}\n"
        "MRR: {mrr}\n"
        "{hits}"
    ).format(
        num_tests=numDatapoints,
        precision=f"{prec:.4f} ± {math.sqrt(prec * (1 - prec) / numDatapoints):.4f}" if numDatapoints > 0 else "N/A",
        recall=f"{rec:.4f} ± {math.sqrt(rec * (1 - rec) / numDatapoints):.4f}" if numDatapoints > 0 else "N/A",
        mrr=f"{getSumOfDictVals(mrr) / numDatapoints:.4f} ± {statistics.stdev(mrr.values()) if len(mrr) > 1 else 0:.4f}" if numDatapoints > 0 else "N/A",
        hits="\n".join(
            [
                "Hits@{}: {:.4f} ± {:.4f}".format(
                    hitIndex + 1,
                    hitVal := sum(
                        [
                            (
                                hitList[hitIndex]
                                if hitIndex < len(hitList)
                                else hitList[-1]
                            )
                            for hitList in hits.values()
                        ]
                    )
                    / numDatapoints,
                    math.sqrt(hitVal * (1 - hitVal) / numDatapoints),
                )
                for hitIndex in range(max(len(hitList) for hitList in hits.values()) if hits else 0)
            ]
        ) if hits else "N/A",
    )


# ============= MAIN =============
parser = argparse.ArgumentParser()
parser.add_argument("--base_model_path", type=str, default="Qwen/Qwen3-0.6B")
parser.add_argument("--num_test_points", type=int, default=0, help="Number of test points (0 = use full dataset)")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
parser.add_argument("--skip_generation", action="store_true", help="Skip response generation (use cached)")
parser.add_argument("--skip_augmentation", action="store_true", help="Skip augmentation phase")
parser.add_argument("--skip_evaluation", action="store_true", help="Skip evaluation phase")
parser.add_argument("--num_shots", type=int, default=0, help="Number of shots to use (default: 0)")
parser.add_argument("--output_augmented_json", type=str, default="augmented_full_dataset.json", help="Output filename for augmented dataset")
args = parser.parse_args()

# Create run directory based on model and shot count
model_name = args.base_model_path.split("/")[-1]  # e.g., "Qwen2.5-32B-Instruct"
run_dir = f"runs/{model_name}_{args.num_shots}shot"
os.makedirs(run_dir, exist_ok=True)

print(f"\n{'='*60}")
print(f"Run Directory: {run_dir}")
print(f"{'='*60}")
print(args)

# Load dataset
print("\nLoading ReDial dataset...")
redial_dataset = load_dataset("community-datasets/re_dial")

full_dataset = []
for split in ["train", "test", "validation"]:
    if split in redial_dataset:
        split_data = redial_dataset[split].to_list()
        print(f"  Loaded {len(split_data)} conversations from {split} split")
        full_dataset.extend(split_data)

print(f"Total conversations in full dataset: {len(full_dataset)}")
random.shuffle(full_dataset)

# Prepare shots and test set
num_shots_needed = max(args.num_shots, 10)
shots = full_dataset[:num_shots_needed]

if args.num_test_points > 0:
    testSet = full_dataset[num_shots_needed : (num_shots_needed + args.num_test_points)]
else:
    testSet = full_dataset[num_shots_needed:]

print(f"\nDataset prepared:")
print(f"  Shots: {args.num_shots}")
print(f"  Test samples: {len(testSet)}")
print(f"  Batch size: {args.batch_size}")
print(f"  Model: {args.base_model_path}")

# Load model if needed for generation
model = None
tokenizer = None
if not args.skip_generation:
    device = "cuda"
    print(f"\nLoading model {args.base_model_path} on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print("Model loaded successfully!")
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

# PHASE 1: Generate responses
if not args.skip_generation:
    responses = generate_responses(testSet, shots[:args.num_shots], model, tokenizer, args, run_dir)
else:
    # Load cached responses
    num_shots_str = str(args.num_shots) + "shot"
    name = args.base_model_path.replace("/", "_") + "_" + num_shots_str
    responsesPath = os.path.join(run_dir, "responses", name + ".json")
    print(f"\nLoading cached responses from {responsesPath}")
    with open(responsesPath, "r") as file:
        responses = json.load(file)

# PHASE 2: Augment dataset
if not args.skip_augmentation:
    augmented_conversations = augment_dataset(testSet, responses, args, run_dir)

# PHASE 3: Evaluate
if not args.skip_evaluation:
    eval_file = os.path.join(run_dir, args.output_augmented_json)
    
    try:
        evaluator = EvaluateAugmentedDataset(stitched_path=eval_file, original_dataset_path=None)
        results = evaluator.evaluate()
        print_results(results)
    except FileNotFoundError as e:
        print(f"\nERROR: Cannot run evaluation. {e}")
        print(f"Please ensure the expected input file exists at: {eval_file}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during evaluation: {e}")