import json
import os
import re
import shutil
from glob import glob

'''
I have added logic to test.py to save recommendations to a directory for later use. 

This script will take the saved recommendations and also do some parenthesis parsing to make sure it is safe to use for triples. It will reference the recommendations and then 
look at the appropriate conversation saved in original_snippets to update the ground truth. It works on initiatorQuestions if the field has data, otherwise uses respondentQuestions.
Currently it is only running on 5 shot results for testing but should do fine with varied number of shots and increasing number of conversations. What it runs on depends on what
test.py was run on. I ran it as python test.py --num_test_points 3 to get 3 conversations. There is added logic in test.py currently to only save 5 shot versions of recommendations. This 
can be removed to get more recommendations and original snippets. 

Naming convention for recommendations is {shot number}shots_{conversation number}.json
Naming convention for original snippets is original_ReDial_snippet_{shot number}shots_{conversation number}.json
Naming convention for augmented datasets is updated_ReDial_from_{shot number}shots_{conversation number}.json

Run the script using python augmentedDataset.py

Currently, the updates are not very robust since it's all dependent on the triples extracted. This means we have a lot of '2s' coming through in the updated dataset.
'''

class AugmentDataset:
    def __init__(self):
        self.conversation = None 
        self.extracted_triples = []

    def load_conversation_from_snippet(self, path):
        """Load the conversation from the original_snippets file"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # The snippet file contains a list with one conversation
        if isinstance(data, list) and len(data) > 0:
            self.conversation = data[0]
        else:
            raise ValueError(f"Invalid snippet format in {path}")
        
        print(f"Loaded conversation from {path}")

    def update_conversation(self):
        """Update the conversation's ground truth with extracted triples"""
        if not self.conversation:
            raise ValueError("Conversation not loaded.")
        if not self.extracted_triples:
            raise ValueError("No triples loaded.")

        # Choose initiatorQuestions, or respondentQuestions if initiator is empty
        answers = (
            self.conversation["initiatorQuestions"]
            if self.conversation.get("initiatorQuestions")
            else self.conversation.get("respondentQuestions", [])
        )

        movie_mentions = {
            m["movieId"]: m["movieName"] for m in self.conversation.get("movieMentions", [])
        }

        print(f"\nDEBUG: Movie mentions in conversation:")
        for movie_id, movie_name in movie_mentions.items():
            print(f"  {movie_id}: {movie_name}")
        
        print(f"\nDEBUG: Extracted triples:")
        for triple in self.extracted_triples:
            print(f"  {triple}")

        # Helper: map movie names in triples to IDs
        def find_movie_id_from_obj(obj_value: str):
            obj_clean = obj_value.strip().lower()
            for movie_id, movie_name in movie_mentions.items():
                if movie_name.strip().lower() in obj_clean:
                    return movie_id
            return None

        # Build triples that reference movie IDs instead of names
        triples_with_ids = []
        for subj, relation, obj in self.extracted_triples:
            movie_id = find_movie_id_from_obj(obj)
            if movie_id:
                triples_with_ids.append((subj.lower(), relation.lower(), movie_id))
            else:
                print(f" No movieId found for triple object: {obj}")

        # Update conversation entries using movieId-based matching
        for answer in answers:
            answer["suggested"] = 2
            answer["seen"] = 2
            answer["liked"] = 2

            movie_id = answer.get("movieId")
            movie_name = movie_mentions.get(movie_id)
            if not movie_id or not movie_name:
                continue

            print(f"\nDEBUG: Checking movie ID {movie_id}: {movie_name}")

            matched_triples = [t for t in triples_with_ids if t[2] == movie_id]

            if matched_triples:
                print(f"  MATCHED RELATIONS: {[r for (_, r, _) in matched_triples]}")

            for _, relation, _ in matched_triples:
                if relation in ("wassuggested", "suggested"):
                    answer["suggested"] = 1
                elif relation == "likes":
                    answer["liked"] = 1
                elif relation in ("dislikes", "hated", "didntlike"):
                    answer["liked"] = 0
                elif relation == "seen":
                    answer["seen"] = 1
                elif relation == "notseen":
                    answer["seen"] = 0

            if not matched_triples:
                print(f"  NO MATCHES found for movie ID {movie_id}")

        print(f"\n Updated conversation with {len(triples_with_ids)} mapped triples")



    def save_updated_conversation(self, path):
        """Save the updated conversation"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump([self.conversation], f, indent=2, ensure_ascii=False)
        print(f"Updated conversation saved to {path}")

    def load_conversation_from_data(self, conversation_obj):
        """Load the conversation from an in-memory object (no file IO)."""
        if isinstance(conversation_obj, dict):
            self.conversation = conversation_obj
        elif isinstance(conversation_obj, list) and len(conversation_obj) > 0 and isinstance(conversation_obj[0], dict):
            # sometimes callers may pass a single-element list containing the conversation
            self.conversation = conversation_obj[0]
        else:
            raise ValueError("Invalid conversation object provided")
        print("Loaded conversation from in-memory object")

    def load_triples_from_list(self, raw_triples):
        """Load extracted triples from an in-memory list (no file IO).

        raw_triples may be a list of strings like "(User, likes, Movie (2017))"
        or a list of 3-tuples/lists.
        """
        converted_triples = []
        for triple in raw_triples:
            if isinstance(triple, str):
                triple_clean = triple.strip()

                # Fix global imbalance first
                open_parens = triple_clean.count("(")
                close_parens = triple_clean.count(")")
                if open_parens > close_parens:
                    triple_clean += ")" * (open_parens - close_parens)
                elif close_parens > open_parens:
                    triple_clean = "(" * (close_parens - open_parens) + triple_clean

                # Remove surrounding parentheses if they wrap the entire triple
                if triple_clean.startswith("(") and triple_clean.endswith(")"):
                    triple_clean = triple_clean[1:-1]

                # Split by comma (into at most 3 parts)
                parts = triple_clean.split(",", 2)

                if len(parts) == 3:
                    subject = parts[0].strip(" ()\"")
                    relation = parts[1].strip(" ()\"")
                    obj = parts[2].strip(" ()\"")

                    # Fix missing parentheses in movie titles "Get Out (2017"
                    if obj.count("(") > obj.count(")"):
                        obj += ")" * (obj.count("(") - obj.count(")"))
                    elif obj.count(")") > obj.count("("):
                        obj = "(" * (obj.count(")") - obj.count("(") ) + obj

                    converted_triples.append([subject, relation, obj])
                else:
                    print(f"Skipping malformed triple (after fix): {triple_clean}")

            elif isinstance(triple, (list, tuple)) and len(triple) == 3:
                subj, rel, obj = triple
                subj, rel, obj = str(subj), str(rel), str(obj)
                if obj.count("(") > obj.count(")"):
                    obj += ")" * (obj.count("(") - obj.count(")"))
                elif obj.count(")") > obj.count("("):
                    obj = "(" * (obj.count(")") - obj.count("(") ) + obj
                converted_triples.append([subj, rel, obj])

            else:
                print(f"Skipping malformed triple: {triple}")

        self.extracted_triples = converted_triples
        print(f"Loaded {len(self.extracted_triples)} valid triple(s) from in-memory list")


