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

random.seed(2025)
"""
CHANGES: 

I have added logic to test.py to save recommendations to a directory for later use. 

There is added logic in test.py currently to only save 5 shot versions of recommendations (the triplets).

Prompt has additions for relations, their definitions, and a blank spot for the schema.


"""
PROMPT = """You are a highly precise information extraction engine. Your task is to analyze the following conversation and extract triples about the user's relationships and preferences for a personal knowledge graph.

The triples must strictly follow the format: (User, [Relation], [Object])

### Allowed Relations:
- Likes: The user explicitly states a positive preference, enjoyment, or favorable opinion about something.
- Dislikes: The user explicitly states a negative preference, dislike, or unfavorable opinion about something.
- Seen: The user explicitly states they have experienced, watched, read, visited, or encountered the object.
- notSeen: The user explicitly states they have NOT experienced, watched, read, visited, or encountered the object.
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
                    "seen" if answerSet["liked"] == 1 else "unseen",
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
parser.add_argument("--num_test_points", type=int, default=300)
parser.add_argument("--do_not_load_model", action=argparse.BooleanOptionalAction)
args = parser.parse_args()
print(args)

# ============= Generate responses =============
if not args.do_not_load_model:
    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path, torch_dtype=torch.float16
    ).to(device)
tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=False)


def runTests(dataset, shots=[], name=None, num_shots=0):
    if name is None:
        name = args.base_model_path + "_" + str(len(shots)) + "shot"
    responses = []
    responsesPath = "responses/" + name + ".json"
    
    # Create recommendations directory if it doesn't exist
    recommendationsDir = "recommendations"
    if not os.path.exists(recommendationsDir):
        os.makedirs(recommendationsDir)
    
    # Create original_snippets directory if it doesn't exist
    originalSnippetsDir = "original_snippets"
    if not os.path.exists(originalSnippetsDir):
        os.makedirs(originalSnippetsDir)
    
    # Path for saving extracted triples
    recommendationsPath = os.path.join(recommendationsDir, f"{num_shots}shots.json")
    
    # Path for saving original conversation snippet
    snippetPath = os.path.join(originalSnippetsDir, f"original_ReDial_snippet_{num_shots}shots.json")
    
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

            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=4096,
            )
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
                0
            ]

            responses.append(response)
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
            if len(hits[index]) < len(recommendations):
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
 # ========REMOVE THIS LATER=====================================================================================   
    # Only save files for 5-shot configuration
    if num_shots == 5:
        print(f"\nSaving {len(all_conversations)} conversations and their extracted triples...")
        
        for conv_index in all_conversations.keys():
            # Save each conversation to original_snippets
            snippet_filename = f"original_ReDial_snippet_{num_shots}shots_{conv_index}.json"
            snippet_path_indexed = os.path.join(originalSnippetsDir, snippet_filename)
            
            with open(snippet_path_indexed, "w", encoding="utf-8") as file:
                json.dump([all_conversations[conv_index]], file, indent=2, ensure_ascii=False)
            
            # Save extracted triples for this conversation
            recommendations_filename = f"{num_shots}shots_{conv_index}.json"
            recommendations_path_indexed = os.path.join(recommendationsDir, recommendations_filename)
            
            triples_as_strings = [
                f"({triple[0]}, {triple[1]}, {triple[2]})" 
                for triple in all_extracted_triples_per_conversation[conv_index]
            ]
            
            with open(recommendations_path_indexed, "w") as file:
                json.dump(triples_as_strings, file, indent=2)
            
            print(f"  Saved conversation {conv_index}: {len(triples_as_strings)} triples")
        
        print(f"Finished saving all {len(all_conversations)} conversations for {num_shots}-shot")
    else:
        print(f"Skipping file save for {num_shots}-shot (only saving 5-shot)")
#=========================================================================================================================
    def getSumOfDictVals(dictionary):
        return sum(dictionary.values())

    return (
        "\nOverall Stats\n"
        "Number of Tests: {num_tests}\n"
        "Precision: {precision}\n"
        "Recall: {recall}\n"
        "MRR: {mrr}\n"
        "{hits}"
    ).format(
        num_tests=numDatapoints,
        precision=(
            str(
                prec := getSumOfDictVals(truePositives)
                / (getSumOfDictVals(truePositives) + getSumOfDictVals(falsePositives))
            )
        )
        + " - "
        + str(math.sqrt(prec * (1 - prec) / numDatapoints)),
        recall=str(
            rec := getSumOfDictVals(truePositives)
            / (getSumOfDictVals(truePositives) + getSumOfDictVals(falseNegatives))
        )
        + " - "
        + str(math.sqrt(rec * (1 - rec) / numDatapoints)),
        mrr=str(getSumOfDictVals(mrr) / numDatapoints)
        + " - "
        + str(statistics.stdev(mrr.values()) if numDatapoints > 1 else 0),
        hits="\n".join(
            [
                "Hits@{}: {} - {}".format(
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
                for hitIndex in range(max(len(hitList) for hitList in hits.values()))
            ]
        ),
    )


movieDataset = load_dataset("community-datasets/re_dial")["test"].to_list()
random.shuffle(movieDataset)

numShots = 10
shots = movieDataset[:numShots]
testSet = movieDataset[numShots : (numShots + args.num_test_points)]

print("Performing 0-Shot Test:")
print(f"0-Shot Scores: {runTests(testSet, num_shots=0)}")
print("Performing 1-Shot Test:")
print(f"1-Shot Scores: {runTests(testSet, shots[:1], num_shots=1)}")
print("Performing 3-Shot Test:")
print(f"3-Shot Scores: {runTests(testSet, shots[:3], num_shots=3)}")
print("Performing 5-Shot Test:")
print(f"5-Shot Scores: {runTests(testSet, shots[:5], num_shots=5)}")
print("Performing 10-Shot Test:")
print(f"10-Shot Scores: {runTests(testSet, shots[:10], num_shots=10)}")