# LLm Triple Extractor Benchmark

# Running the benchmark

First, install the dependencies from the `requirements.txt`.

The benchmark can be run with `test.py`.
- You can pass the model with the `--base_model_path` argument (defaults to `Qwen/Qwen3-0.6B`).
- You can set the number of test points to collect by passing a number to `--num_test_points` (defaults to `300`).
- You can tell the script not to load the model by passing `--do_not_load_model`. This is useful if you just want to recalculate the metric from already generated responses (automatically stored in a `responses` folder when you run the script initially).

The list of test data points are shuffled, and then the first `num_test_points` are used for testing, so the chosen test data points are always consistent regardless of the number of test data points, so you can initially test with a small number of test data points, and then rerun the benchmark with a larger number of test data points without having to regenerate the responses for the test data points you already did.

The results are printed to stdout, so I recommend sending it to a file.

Here is an example run:

`python test.py --base_model_path Qwen/Qwen3-8B --num_test_points 100 > Qwen3_8B.out`
