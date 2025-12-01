import json
import os
import re
import shutil
from glob import glob
import urllib.parse

'''
Enhanced version with robust fuzzy name matching.
Uses URL decoding and whitespace normalization for reliable movie name matching.

This script takes saved recommendations and updates ground truth.

Naming conventions:
- Recommendations: {shot_number}shots_{conversation_number}.json
- Snippets: original_ReDial_snippet_{shot_number}shots_{conversation_number}.json
- Output: updated_ReDial_from_{shot_number}shots_{conversation_number}.json

Usage: python augmentDataset.py
'''

class AugmentDataset:
    def __init__(self):
        self.conversation = None 
        self.extracted_triples = []

    def normalize_movie_name(self, name):
        """Normalize movie names for robust fuzzy matching"""
        if not name:
            return ""
        name = urllib.parse.unquote(name)  # Handle URL encoding (%3F -> ?)
        name = name.lower().strip()
        name = re.sub(r'\s+', ' ', name)  # Normalize multiple spaces to single space
        return name

    def load_conversation_from_snippet(self, path):
        """Load the conversation from the original_snippets file"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
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

        # Build lookup: movieId -> list of relations using fuzzy name matching
        movie_relations = {}
        
        for subj, relation, obj in self.extracted_triples:
            movie_id = None
            
            # Fuzzy name matching with normalization
            obj_normalized = self.normalize_movie_name(obj)
            
            for mid, mname in movie_mentions.items():
                mname_normalized = self.normalize_movie_name(mname)
                
                # Bidirectional substring matching
                if mname_normalized in obj_normalized or obj_normalized in mname_normalized:
                    movie_id = mid
                    print(f"  ✓ Matched ID {movie_id}: '{obj}' -> '{mname}'")
                    break
            
            if movie_id:
                if movie_id not in movie_relations:
                    movie_relations[movie_id] = []
                movie_relations[movie_id].append(relation.lower())
            else:
                print(f"  ✗ No movieId found for triple object: {obj}")

        # Update conversation entries using exact movieId matching
        for answer in answers:
            answer["suggested"] = 2
            answer["seen"] = 2
            answer["liked"] = 2

            movie_id = answer.get("movieId")
            movie_name = movie_mentions.get(movie_id)
            if not movie_id or not movie_name:
                continue

            print(f"\nDEBUG: Checking movie ID {movie_id}: {movie_name}")

            relations = movie_relations.get(movie_id, [])

            if relations:
                print(f"  MATCHED RELATIONS: {relations}")

                for relation in relations:
                    if relation in ("wassuggested", "suggested"):
                        answer["suggested"] = 1
                    elif relation == "likes":
                        answer["liked"] = 1
                    elif relation in ("dislikes", "hated", "didntlike", "disliked"):
                        answer["liked"] = 0
                    elif relation == "seen":
                        answer["seen"] = 1
                    elif relation in ("notseen", "unseen"):
                        answer["seen"] = 0
            else:
                print(f"  NO MATCHES found for movie ID {movie_id}")

        print(f"\nUpdated conversation with {len(self.extracted_triples)} mapped triples")

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
            self.conversation = conversation_obj[0]
        else:
            raise ValueError("Invalid conversation object provided")
        print("Loaded conversation from in-memory object")

    def load_triples_from_list(self, raw_triples):
        """Load extracted triples with robust parsing.
        
        Expected format: "(User, likes, Movie Name (Year))"
        """
        converted_triples = []
        
        for triple in raw_triples:
            if isinstance(triple, str):
                triple_clean = triple.strip()

                # Fix global parenthesis imbalance
                open_parens = triple_clean.count("(")
                close_parens = triple_clean.count(")")
                if open_parens > close_parens:
                    triple_clean += ")" * (open_parens - close_parens)
                elif close_parens > open_parens:
                    triple_clean = "(" * (close_parens - open_parens) + triple_clean

                # Remove outer parentheses wrapping the entire triple
                if triple_clean.startswith("(") and triple_clean.endswith(")"):
                    triple_clean = triple_clean[1:-1]

                # Split by comma (max 3 parts to handle movie titles with commas)
                parts = triple_clean.split(",", 2)

                if len(parts) == 3:
                    subject = parts[0].strip(" ()\"")
                    relation = parts[1].strip(" ()\"")
                    obj = parts[2].strip(" ()\"")

                    # Fix missing parentheses in movie titles like "Get Out (2017"
                    if obj.count("(") > obj.count(")"):
                        obj += ")" * (obj.count("(") - obj.count(")"))
                    elif obj.count(")") > obj.count("("):
                        obj = "(" * (obj.count(")") - obj.count("(")) + obj

                    converted_triples.append([subject, relation, obj])
                else:
                    print(f"Skipping malformed triple (wrong number of parts): {triple_clean}")

            elif isinstance(triple, (list, tuple)) and len(triple) == 3:
                subj, rel, obj = triple
                subj, rel, obj = str(subj), str(rel), str(obj)
                
                # Fix parenthesis imbalance
                if obj.count("(") > obj.count(")"):
                    obj += ")" * (obj.count("(") - obj.count(")"))
                elif obj.count(")") > obj.count("("):
                    obj = "(" * (obj.count(")") - obj.count("(")) + obj
                
                converted_triples.append([subj, rel, obj])
            else:
                print(f"Skipping malformed triple: {triple}")

        self.extracted_triples = converted_triples
        print(f"Loaded {len(self.extracted_triples)} valid triple(s) from in-memory list")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Augment ReDial conversations using extracted triples with fuzzy name matching"
    )
    parser.add_argument(
        "--recommendations_dir",
        type=str,
        default="recommendations",
        help="Directory containing recommendation JSON files"
    )
    parser.add_argument(
        "--snippets_dir",
        type=str,
        default="original_snippets",
        help="Directory containing original snippet JSON files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="augmented_conversations",
        help="Directory to save augmented conversations"
    )
    parser.add_argument(
        "--num_shots",
        type=str,
        default="1shot",
        help="Shot configuration to process (e.g., '1shot', '5shot')"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed debug information"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("ReDial Dataset Augmentation - Fuzzy Name Matching")
    print("="*60)
    print(f"Recommendations dir: {args.recommendations_dir}")
    print(f"Snippets dir: {args.snippets_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Shot configuration: {args.num_shots}")
    print("="*60)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    recommendation_pattern = os.path.join(
        args.recommendations_dir, 
        f"{args.num_shots}_*.json"
    )
    recommendation_files = sorted(glob(recommendation_pattern))
    
    if not recommendation_files:
        print(f"\nERROR: No files found matching: {recommendation_pattern}")
        print("Run test.py with --save_for_augmentation first")
        exit(1)
    
    print(f"\nFound {len(recommendation_files)} recommendation files")
    
    num_processed = 0
    num_failed = 0
    num_no_triples = 0
    
    for rec_file in recommendation_files:
        try:
            basename = os.path.basename(rec_file)
            index = basename.replace(f"{args.num_shots}_", "").replace(".json", "")
            
            snippet_file = os.path.join(
                args.snippets_dir,
                f"original_ReDial_snippet_{args.num_shots}_{index}.json"
            )
            
            if not os.path.exists(snippet_file):
                print(f"Warning: Snippet not found: {snippet_file}")
                num_failed += 1
                continue
            
            with open(rec_file, "r", encoding="utf-8") as f:
                triples = json.load(f)
            
            if not triples:
                if args.verbose:
                    print(f"No triples in {basename}")
                num_no_triples += 1
                continue
            
            augmenter = AugmentDataset()
            augmenter.load_conversation_from_snippet(snippet_file)
            augmenter.load_triples_from_list(triples)
            
            if not args.verbose:
                import sys
                from io import StringIO
                old_stdout = sys.stdout
                sys.stdout = StringIO()
            
            augmenter.update_conversation()
            
            if not args.verbose:
                sys.stdout = old_stdout
            
            output_file = os.path.join(
                args.output_dir,
                f"updated_ReDial_from_{args.num_shots}_{index}.json"
            )
            augmenter.save_updated_conversation(output_file)
            
            num_processed += 1
            
            if args.verbose:
                print(f"✓ Processed conversation {index}")
            
        except Exception as e:
            print(f"\nError processing {rec_file}: {e}")
            num_failed += 1
    
    print(f"\n{'='*60}")
    print("Processing Complete!")
    print(f"{'='*60}")
    print(f"Successfully processed: {num_processed}")
    print(f"No triples found: {num_no_triples}")
    print(f"Failed: {num_failed}")
    print(f"{'='*60}")