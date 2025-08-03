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

# PROMPT = '''You will analyze the following conversation to extract triples about the user for a personal knowledge graph about them.

# ```json
# {}
# ```

# The triples should be in the following format:
# (User, [Relation], [Object])

# These are the following relations to consider:
# Likes: Indicates that the user likes the object
# Dislikes: Indicates that the user dislikes the object
# Seen: Indicates that the user has seen the object
# Unseen: Indicates that the user has not seen the object
# Suggested: Indicates that the object has been recommended to the user
# '''

PROMPT = """You are a highly precise information extraction engine. Your task is to analyze the following conversation and extract triples about the user's relationship with movies for a personal knowledge graph.

The triples must strictly follow the format: (User, [Relation], [Object])

### Allowed Relations:
- Likes: The user explicitly states a positive preference for a movie or movie genre.
- Dislikes: The user explicitly states a negative preference for a movie or movie genre.
- Seen: The user explicitly states they have watched the movie.
- Unseen: The user explicitly states they have NOT watched the movie.
- Suggested: A movie is recommended to the user by the assistant.

### Rules:
1. The `[Object]` of the triple must be the title of a specific movie or a genre of movies. Do not extract other types of objects like actors, directors, or general topics.
2. The `[Object]` should be as specific as possible (include the whole name).
3. If no relevant triples can be extracted from the conversation, output the text: "No triples found."
4. Focus only on triples where the user is the subject.
5. Provide triples for all applicable relations.
{}
### Your Task

**Input Conversation:**
```json
{}
```"""

TRIPLE_FORMAT = "({}, {}, {})"

SHOT_TEMPLATE = """---

### Example Task

**Input Conversation:**
```json
{}
```

**Correct Output:**
{}

---"""


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


def runTests(dataset, shots=[], name=None):
    if name is None:
        name = args.base_model_path + "_" + str(len(shots)) + "shot"
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
    for index, dataPoint in enumerate(tqdm(dataset)):
        if saveResponses or  index >= len(responses):
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
        + str(statistics.stdev(mrr.values())),
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
print(f"0-Shot Scores: {runTests(testSet)}")
print("Performing 1-Shot Test:")
print(f"1-Shot Scores: {runTests(testSet, shots[:1])}")
print("Performing 3-Shot Test:")
print(f"3-Shot Scores: {runTests(testSet, shots[:3])}")
print("Performing 5-Shot Test:")
print(f"5-Shot Scores: {runTests(testSet, shots[:5])}")
print("Performing 10-Shot Test:")
print(f"10-Shot Scores: {runTests(testSet, shots[:10])}")
