import argparse
import json
import math
import sys
import re
import random
from collections import defaultdict
import os
import torch
import sys
import statistics

from tqdm import tqdm
import datasets
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import shutil
from glob import glob

from augmentDataset import AugmentDataset
from evaluateDataset import EvaluateAugmentedDataset

random.seed(2025)
"""
CHANGES: 

I have added logic to test.py to save recommendations to a directory for later use. 

There is added logic in test.py currently to only save 5 shot versions of recommendations (the triplets).

Prompt has additions for relations, their definitions, and a blank spot for the schema.

Added support for any size Qwen model with proper memory management and device handling. Were some errors before that were causing some crashes on larger models.
"""
PROMPT = """You are a highly precise information extraction engine. Your task is to analyze the following conversation and extract triples about the user's relationships and preferences for a personal knowledge graph.

The triples must strictly follow the format: (User, [Relation], [Object])

### Allowed Relations:
- Likes: The user expresses positive preference, enjoyment, or favorable opinion about something.
- Dislikes: The user expresses negative preference, dislike, or unfavorable opinion about something.
- Seen: The user indicates they have experienced, watched, read, visited, or encountered the object.
- notSeen: The user indicates they have NOT experienced, watched, read, visited, or encountered the object.
- wasSuggested: An item, activity, or topic is recommended to the user by the assistant.

### Rules:
1. The `[Object]` can be any entity, item, topic, activity, place, or concept discussed in the conversation (e.g., movies, books, restaurants, hobbies, products, places, etc.).
2. The `[Object]` should be as specific as possible (include the complete name or detailed description).
3. If no relevant triples can be extracted from the conversation, output the text: "No triples found."
4. Focus only on triples where the user is the subject.
5. Provide triples for all applicable relations.
6. Extract triples for any domain or topic - not limited to a specific category.
{}

### Schema:
{}

### Your Task

**Input Conversation:**
```json
{}
```"""

TRIPLE_FORMAT = "({}, {}, {})"

SHOT_TEMPLATE = """### Example Task

**Input Conversation:**
```json
{}
```

**Correct Output:**
{}"""


def messagesToConversation(datapoint):
    messages = datapoint["messages"]
    initiatorID = datapoint["initiatorWorkerId"]

    def replaceMovieMentions(message):
        for movieMention in datapoint["movieMentions"]:
            message = message.replace(
                "@" + movieMention["movieId"], movieMention["movieName"]
            )
        return message

    return json.dumps(
        [
            {
                "content": replaceMovieMentions(message["text"]),
                "role": (
                    "user" if message["senderWorkerId"] == initiatorID else "assistant"
                ),
            }
            for message in messages
        ],
        indent=2,
    )


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
                    "likes" if answerSet["liked"] == 1 else "disliked",
                    movieMentions[answerSet["movieId"]],
                )
            )
        if not (answerSet["seen"] == 2):
            triples.append(
                (
                    "User",
                    "seen" if answerSet["seen"] == 1 else "unseen",
                    movieMentions[answerSet["movieId"]],
                )
            )
        if not (answerSet["suggested"] == 1):
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
    return "\n---\n\n" + "".join([(createShot(shot) + "\n\n---\n") for shot in shots])


def extractTriplesFromResponse(response):
    """Extract triples from model response and return as list of tuples"""
    recommendations = re.findall(r"\([^,]+, [^,]+, [^\)]+\)?\)", response)
    extracted = []
    
    for rec in recommendations:
        # Parse the triple string into components
        # Remove parentheses and split by comma
        rec_clean = rec.strip("()")
        parts = [p.strip() for p in rec_clean.split(",")]
        if len(parts) == 3:
            extracted.append(tuple(parts))
    
    return extracted


parser = argparse.ArgumentParser()
parser.add_argument("--base_model_path", type=str, default="Qwen/Qwen3-0.6B")
parser.add_argument("--num_test_points", type=int, default=None)
parser.add_argument("--do_not_load_model", action=argparse.BooleanOptionalAction)
parser.add_argument("--device", type=str, default="auto", help="Device to use: 'cuda', 'cpu', or 'auto'")
parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit precision for memory efficiency")
parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit precision for maximum memory efficiency")
args = parser.parse_args()
print(args)

# ============= Setup device and model loading config =============
if args.device == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = args.device

print(f"Using device: {device}")

# ============= Generate responses =============
if not args.do_not_load_model:
    model_kwargs = {
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        "low_cpu_mem_usage": True,
    }
    
    # Add quantization if requested (requires bitsandbytes)
    if args.load_in_8bit:
        model_kwargs["load_in_8bit"] = True
        model_kwargs["device_map"] = "auto"
        print("Loading model in 8-bit precision")
    elif args.load_in_4bit:
        model_kwargs["load_in_4bit"] = True
        model_kwargs["device_map"] = "auto"
        print("Loading model in 4-bit precision")
    else:
        # For non-quantized models, handle device placement manually
        if device == "cuda":
            # Check available GPU memory and decide on device_map
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # in GB
                print(f"Available GPU memory: {gpu_memory:.2f} GB")
                
                # For larger models (>3B params), use device_map="auto" for better memory management
                model_kwargs["device_map"] = "auto"
        
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path, 
            **model_kwargs
        )
        
        # Only move to device if not using device_map
        if "device_map" not in model_kwargs:
            model = model.to(device)
            
        print(f"Model loaded successfully on {device}")
        
        # Enable gradient checkpointing for larger models to save memory
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Try using --load_in_8bit or --load_in_4bit flags for large models")
        sys.exit(1)
else:
    model = None

tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=False)

# Set pad token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def runTests(dataset, shots=[], name=None, num_shots=0):
    if name is None:
        name = args.base_model_path.replace("/", "_") + "_" + str(len(shots)) + "shot"
    responses = []
    responsesPath = "responses/" + name + ".json"
    
    saveResponses = True
    if os.path.exists(responsesPath):
        with open(responsesPath, "r") as file:
            responses = json.load(file)
        saveResponses = False
    else:
        responsesFolder = responsesPath[: responsesPath.rfind("/")]
        if not os.path.exists(responsesFolder):
            os.makedirs(responsesFolder)

    truePositives = defaultdict(int)
    falsePositives = defaultdict(int)
    falseNegatives = defaultdict(int)
    hits = defaultdict(lambda: [0 for _ in range(10)])
    mrr = defaultdict(int)
    numDatapoints = 0
    
    # Store extracted triples and conversations for ALL datapoints
    all_extracted_triples_per_conversation = {}
    all_conversations = {}
    
    for index, dataPoint in enumerate(tqdm(dataset)):
        # Save all conversations
        all_conversations[index] = dataPoint
        
        if saveResponses or index >= len(responses):
            saveResponses = True
            text = tokenizer.apply_chat_template(
                [
                    {
                        "content": PROMPT.format(
                            createShots(shots), "", messagesToConversation(dataPoint)
                        ),
                        "role": "user",
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )

            print(f"---------------- PROMPT --------------\n{text}")

            # Use model.device if available, otherwise use the device variable
            model_device = model.device if hasattr(model, 'device') else device
            model_inputs = tokenizer([text], return_tensors="pt").to(model_device)

            # Add generation config for better stability across model sizes
            generation_config = {
                "max_new_tokens": 4096,
                "do_sample": False,  # Use greedy decoding for consistency
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }

            with torch.no_grad():  # Disable gradients for inference
                generated_ids = model.generate(
                    **model_inputs,
                    **generation_config,
                )
            
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            responses.append(response)
            
            # Clear CUDA cache periodically to prevent OOM
            if device == "cuda" and index % 10 == 0:
                torch.cuda.empty_cache()
        else:
            response = responses[index]
            
        print(f"---------------- RESPONSE --------------\n{response}")
        
        if "Qwen" in args.base_model_path:
            endThinkString = "</think>"
            endThinkIndex = response.rfind(endThinkString)
            if endThinkIndex == -1:
                print("Output did not complete thinking.")
                continue
        if len(tokenizer.encode(response, add_special_tokens=True)) > 4090:
            print("Output did not complete.")
            continue
        if "Qwen" in args.base_model_path:
            response = response[(endThinkIndex + len(endThinkString)) :]

        goals = getTriples(dataPoint)
        print(f"---------------- GOALS --------------\n{goals}")

        # Extract triples from response for datapoints
        extracted_triples = extractTriplesFromResponse(response)
        all_extracted_triples_per_conversation[index] = extracted_triples
        print(f"---------------- EXTRACTED TRIPLES --------------\n{extracted_triples}")

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
                        foundGoal = (
                            recommendationSubstring.find(
                                goal[2][: goal[2].find(" (")].lower()
                            )
                            >= 0
                        )
                if foundGoal:
                    goalNotFound = False
                    truePositives[index] += 1
                    print(f"{goal} found in response.")
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
                print(f"{goal} not found in response.")
            if doBreak:
                break
        falseNegatives[index] = max(
            0, min(20 - truePositives[index], falseNegatives[index])
        )
        mrr[index] = 0
        if rank >= 0:
            mrr[index] += 1 / (rank + 1)
            if index not in hits:
                hits[index] = [0] * len(recommendations)
            elif len(hits[index]) < len(recommendations):
                hits[index] += [hits[index][-1]] * (
                    len(recommendations) - len(hits[index])
                )
            for i in range(rank, len(hits[index]), 1):
                hits[index][i] += 1
        numDatapoints += 1
        print(f"truePositives[index]: {truePositives[index]}")
        print(f"falsePositives[index]: {falsePositives[index]}")
        print(f"falseNegatives[index]: {falseNegatives[index]}")
        print(f"Hits@: {hits[index]}")

    if saveResponses:
        with open(responsesPath, "w") as file:
            json.dump(responses, file)
            
    if num_shots == 5:
        print(f"Collected {len(all_conversations)} conversations and their extracted triples in-memory for {num_shots}-shot")
    else:
        print(f"Collected {len(all_conversations)} conversations (not persisting snippets/recommendations to disk)")
    
    def getSumOfDictVals(dictionary):
        return sum(dictionary.values())

    tp = getSumOfDictVals(truePositives)
    fp = getSumOfDictVals(falsePositives)
    fn = getSumOfDictVals(falseNegatives)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    stats_str = (
        "\nOverall Stats\n"
        "Number of Tests: {num_tests}\n"
        "Precision: {precision}\n"
        "Recall: {recall}\n"
        "MRR: {mrr}\n"
        "{hits}"
    ).format(
        num_tests=numDatapoints,
        precision=(
            str(prec)
            + " - "
            + str(math.sqrt(prec * (1 - prec) / numDatapoints) if numDatapoints > 0 else 0.0)
        ),
        recall=(
            str(rec)
            + " - "
            + str(math.sqrt(rec * (1 - rec) / numDatapoints) if numDatapoints > 0 else 0.0)
        ),
        mrr=(
            str(getSumOfDictVals(mrr) / numDatapoints if numDatapoints > 0 else 0)
            + " - "
            + str(statistics.stdev(list(mrr.values())) if len(mrr) > 1 else 0)
        ),
        hits=(
            "\n".join([
                "Hits@{}: {} - {}".format(
                    hitIndex + 1,
                    hitVal := (
                        sum([
                            (hitList[hitIndex] if hitIndex < len(hitList) else hitList[-1])
                            for hitList in hits.values()
                        ]) / numDatapoints if numDatapoints > 0 else 0
                    ),
                    math.sqrt(hitVal * (1 - hitVal) / numDatapoints) if numDatapoints > 0 else 0,
                )
                for hitIndex in range(max((len(hitList) for hitList in hits.values()), default=0))
            ]) if hits else "No hits data"
        ),
    )
    
    return (
        stats_str,
        all_conversations,
        all_extracted_triples_per_conversation,
    )

if __name__ == "__main__":
    movieDataset = load_dataset("community-datasets/re_dial")["test"].to_list()
    random.shuffle(movieDataset)

    numShots = 10
    shots = movieDataset[:numShots]
    if args.num_test_points:
        testSet = movieDataset[numShots : (numShots + args.num_test_points)]
    else:
        testSet = movieDataset[numShots:]

    print("Performing 5-Shot Test:")
    stats_str, all_conversations, all_extracted_triples_per_conversation = runTests(
        testSet, shots[:5], num_shots=5
    )
    print(f"5-Shot Scores:\n{stats_str}")

    stitched = []
    kept_indices = []  
    processed_count = 0

    # Sort by the original index (ReDial order)
    for index in sorted(all_conversations.keys()):
        conv = all_conversations[index]
        triples = all_extracted_triples_per_conversation.get(index, [])

        processor = AugmentDataset()
        processor.load_conversation_from_data(conv)
        processor.load_triples_from_list(triples)

        if not processor.extracted_triples:
            continue

        processor.update_conversation()
        augmented_conv = processor.conversation
        
        if "conversationId" in conv:
            augmented_conv["_original_conversation_id"] = conv["conversationId"]
        augmented_conv["_original_index"] = index
        
        stitched.append(augmented_conv)
        kept_indices.append(index) 
        processed_count += 1

    print(f"\n{'='*60}")
    print(f"Processing complete! Processed {processed_count} conversations.")
    print(f"{'='*60}")

    OUTPUT_FILE = "stitched_conversations.json"
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(stitched, f, indent=2, ensure_ascii=False)

    print(f"\nStitched {len(stitched)} conversations into {OUTPUT_FILE}")

    filtered_testSet = [testSet[i] for i in kept_indices]
    with open("test_set.json", "w", encoding="utf-8") as f:
        json.dump(filtered_testSet, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(filtered_testSet)} filtered conversations to test_set.json for evaluation")

    evaluator = EvaluateAugmentedDataset(
        stitched_path="stitched_conversations.json"
    )
    metrics = evaluator.evaluate()