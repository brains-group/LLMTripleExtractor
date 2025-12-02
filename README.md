# LLm Triple Extractor Benchmark

## Wikidata Movie Data Extraction

The `wikidata_query` subfolder contains a script to query Wikidata for movie data. The extracted data is saved to the `sample_data` folder, which contains sample data for reproducibility purposes.

### Querying Movies from Wikidata

The `wikidata_query/query_movies.py` script queries the Wikidata SPARQL endpoint to retrieve movie titles, their IRIs (Internationalized Resource Identifiers), and release years. The script automatically deduplicates entries to ensure each unique combination of (title, IRI, year) appears only once in the output.

**Usage:**

```bash
# Query all movies (may take a long time - there are many movies in Wikidata)
# Output will be saved to sample_data/movies.csv by default
python wikidata_query/query_movies.py

# Query with a limit (recommended for testing)
python wikidata_query/query_movies.py --limit 1000

# Specify a custom output path
python wikidata_query/query_movies.py --output sample_data/movies.csv --limit 1000
```

**Arguments:**
- `--output`: Path to the output CSV file (default: `sample_data/movies.csv`)
- `--limit`: Maximum number of movies to retrieve (default: all movies)

**Output Format:**

The CSV file contains three columns:
- `title`: Movie title
- `iri`: Wikidata IRI (e.g., `http://www.wikidata.org/entity/Q12345`)
- `year`: Release year (extracted from publication date)

The script uses pagination to handle large queries efficiently and automatically removes duplicate entries based on the combination of title, IRI, and year.

## Running the benchmark

First, install the dependencies from the `requirements.txt`.

The benchmark can be run with `test.py`.
- You can pass the model with the `--base_model_path` argument (defaults to `Qwen/Qwen3-0.6B`).
- You can set the number of test points to collect by passing a number to `--num_test_points` (defaults to `300`).
- You can tell the script not to load the model by passing `--do_not_load_model`. This is useful if you just want to recalculate the metric from already generated responses (automatically stored in a `responses` folder when you run the script initially).

The list of test data points are shuffled, and then the first `num_test_points` are used for testing, so the chosen test data points are always consistent regardless of the number of test data points, so you can initially test with a small number of test data points, and then rerun the benchmark with a larger number of test data points without having to regenerate the responses for the test data points you already did.

The results are printed to stdout, so I recommend sending it to a file.

Here is an example run:

`python test.py --base_model_path Qwen/Qwen3-8B --num_test_points 100 > Qwen3_8B.out`
