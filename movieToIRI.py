import csv
import json
import re
from fuzzywuzzy import fuzz

# Paths
triples_file = "recommendations/5shots_0.json"       # JSON file containing triples
dataset_file = "movies.csv"  # CSV file containing title, iri, year

# Load triples
with open(triples_file, "r", encoding="utf-8") as f:
    triples = json.load(f)

# Load dataset
movies_data = []
with open(dataset_file, newline="", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # skip header
    for row in reader:
        title, iri, year = row
        movies_data.append({
            "title": title.strip('"'),
            "iri": iri,
            "year": year
        })

# Parse triple to extract movie name and year
def parse_triple(triple):
    # Match 4-digit year at end (allow extra closing parentheses)
    match = re.search(r'\((\d{4})\)\)*\s*$', triple)
    if match:
        year = match.group(1)
        name_part = triple[:match.start()]
        name = name_part.split(",")[-1].strip()
        return name, year
    return triple, None

# Thresholds for fuzzy matching
TITLE_THRESHOLD = 90  # require very high similarity for title

# Create mapping dictionary
movie_to_iri = {}

for triple in triples:
    movie_name, movie_year = parse_triple(triple)
    best_score = 0
    best_match = None

    for data in movies_data:
        # Only consider entries with the same year
        if movie_year != data["year"]:
            continue

        title_score = fuzz.token_sort_ratio(movie_name.lower(), data["title"].lower())

        if title_score > best_score:
            best_score = title_score
            best_match = data["iri"]

    # Only accept match if title similarity exceeds threshold
    key = f"{movie_name} ({movie_year})" if movie_year else movie_name
    if best_match and best_score >= TITLE_THRESHOLD:
        movie_to_iri[key] = best_match
    else:
        movie_to_iri[key] = None

# Print result
for k, v in movie_to_iri.items():
    print(k, "->", v)

print(movie_to_iri)
