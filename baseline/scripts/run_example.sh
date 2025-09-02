if [ ! -d "./annotated_data/git" ]; then
  echo "unzip git data..."
  bash ./utils/get_git_dir.sh
fi

# 删除文件
rm -f ./baseline/output/example_results.json

uv run -m baseline.run_conversation --input_path ./baseline/data/example_queries.json --output_path ./baseline/output/example_results.json --max_tools 50 --insert_number 0