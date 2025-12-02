"""
Query Wikidata for movie titles, IRIs, and release years.

This script queries the Wikidata SPARQL endpoint to retrieve all movies
along with their IRIs and release years, then exports the data to a CSV file.
"""

import requests
import csv
import argparse
import os
from typing import List, Dict


def query_wikidata_movies(limit: int = None) -> List[Dict[str, str]]:
    """
    Query Wikidata SPARQL endpoint for movies with their titles, IRIs, and release years.
    
    Args:
        limit: Maximum number of results to retrieve. If None, retrieves all results.
    
    Returns:
        List of dictionaries containing movie data (title, iri, year)
    """
    # SPARQL query to get movies with their labels, IRIs, and publication dates
    # Note: Removed ORDER BY for performance - we'll sort in Python if needed
    base_query = """
    SELECT ?movie ?movieLabel ?publicationDate WHERE {
      ?movie wdt:P31 wd:Q11424 .  # Instance of film
      OPTIONAL { ?movie wdt:P577 ?publicationDate . }  # Publication date (release date)
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
    }
    """
    
    # Use pagination for large queries to avoid timeouts
    page_size = 10000  # Wikidata typically allows up to 10k results per query
    movies = []
    offset = 0
    
    if limit and limit < page_size:
        # Small query, no pagination needed
        query = base_query + f" LIMIT {limit}"
        movies = _execute_query(query)
    else:
        # Large query, use pagination
        while True:
            current_limit = page_size
            if limit:
                remaining = limit - len(movies)
                if remaining <= 0:
                    break
                current_limit = min(page_size, remaining)
            
            query = base_query + f" LIMIT {current_limit} OFFSET {offset}"
            print(f"Fetching movies {offset} to {offset + current_limit}...")
            
            page_results = _execute_query(query)
            if not page_results:
                break
            
            movies.extend(page_results)
            offset += len(page_results)
            
            # If we got fewer results than requested, we've reached the end
            if len(page_results) < current_limit:
                break
            
            # If we've reached the limit, stop
            if limit and len(movies) >= limit:
                movies = movies[:limit]
                break
    
    # Deduplicate: keep only unique combinations of (title, iri, year)
    seen = set()
    unique_movies = []
    for movie in movies:
        # Create a unique key from title, iri, and year
        key = (movie["title"], movie["iri"], movie["year"])
        if key not in seen:
            seen.add(key)
            unique_movies.append(movie)
    
    # Sort by title in Python
    unique_movies.sort(key=lambda x: x["title"].lower())
    
    print(f"Removed {len(movies) - len(unique_movies)} duplicate entries")
    
    return unique_movies


def _execute_query(query: str) -> List[Dict[str, str]]:
    """Execute a single SPARQL query and return results."""
    # Wikidata SPARQL endpoint
    url = "https://query.wikidata.org/sparql"
    
    # Set headers for the request
    headers = {
        "User-Agent": "LLMTripleExtractor/1.0 (https://github.com/yourusername/LLMTripleExtractor)",
        "Accept": "application/sparql-results+json"
    }
    
    # Prepare the request
    params = {
        "query": query,
        "format": "json"
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=300)
        response.raise_for_status()
        
        data = response.json()
        
        movies = []
        bindings = data.get("results", {}).get("bindings", [])
        
        for binding in bindings:
            movie_iri = binding.get("movie", {}).get("value", "")
            movie_label = binding.get("movieLabel", {}).get("value", "")
            publication_date = binding.get("publicationDate", {}).get("value", "")
            
            # Extract year from publication date (format: YYYY-MM-DD or YYYY)
            year = ""
            if publication_date:
                # Publication dates are typically in format YYYY-MM-DD or YYYY
                year = publication_date.split("-")[0] if "-" in publication_date else publication_date
            
            movies.append({
                "title": movie_label,
                "iri": movie_iri,
                "year": year
            })
        
        return movies
    
    except requests.exceptions.RequestException as e:
        print(f"Error querying Wikidata: {e}")
        raise
    except Exception as e:
        print(f"Error processing results: {e}")
        raise


def save_to_csv(movies: List[Dict[str, str]], output_file: str):
    """
    Save movie data to a CSV file.
    
    Args:
        movies: List of dictionaries containing movie data
        output_file: Path to the output CSV file
    """
    if not movies:
        print("No movies to save.")
        return
    
    fieldnames = ["title", "iri", "year"]
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(movies)
    
    print(f"Saved {len(movies)} movies to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Query Wikidata for movies and export to CSV"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sample_data/movies.csv",
        help="Output CSV file path (default: sample_data/movies.csv)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of movies to retrieve (default: all)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Wikidata Movie Query Tool")
    print("=" * 60)
    
    print("Querying Wikidata SPARQL endpoint...")
    if args.limit:
        print(f"Limit: {args.limit} movies")
    else:
        print("Warning: No limit specified. This may retrieve a very large number of movies.")
        print("Consider using --limit to restrict the number of results.")
    
    # Query Wikidata
    movies = query_wikidata_movies(limit=args.limit)
    
    print(f"Retrieved {len(movies)} movies from Wikidata")
    
    # Save to CSV
    save_to_csv(movies, args.output)
    
    print("=" * 60)
    print("Query complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

