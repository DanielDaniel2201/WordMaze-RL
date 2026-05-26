---
dataset_info:
  config_name: m3-4
  features:
  - name: id
    dtype: string
  - name: start
    dtype: string
  - name: goal
    dtype: string
  - name: word_length
    dtype: int64
  - name: max_moves
    dtype: int64
  - name: password
    dtype: string
  - name: prompt
    dtype: string
  - name: messages
    list:
    - name: role
      dtype: string
    - name: content
      dtype: string
  - name: solution
    dtype: string
  - name: answer
    dtype: string
  - name: path
    list: string
  - name: num_moves
    dtype: int64
  splits:
  - name: train
    num_bytes: 2219404
    num_examples: 1600
  - name: validation
    num_bytes: 277562
    num_examples: 200
  - name: test
    num_bytes: 277043
    num_examples: 200
  download_size: 2466944
  dataset_size: 2774009
configs:
- config_name: m3-4
  data_files:
  - split: train
    path: m3-4/train-*
  - split: validation
    path: m3-4/validation-*
  - split: test
    path: m3-4/test-*
---
