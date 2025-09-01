if [ ! -d "./annotated_data/git" ]; then
  echo "unzip git data..."
  bash ./utils/get_git_dir.sh
fi

rm -f ./baseline/output/baseline_results.json

uv run -m baseline.run_conversation --input_path ./annotated_data/all_annotations.json --output_path ./baseline/output/baseline_results.json --max_tools 10 --insert_number 0

