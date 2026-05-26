---
dataset_info:
- config_name: m3-4
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
    num_bytes: 1788587
    num_examples: 1600
  - name: validation
    num_bytes: 223612
    num_examples: 200
  - name: test
    num_bytes: 223810
    num_examples: 200
  download_size: 1924801
  dataset_size: 2236009
- config_name: m4-6
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
    num_bytes: 1861091
    num_examples: 1600
  - name: validation
    num_bytes: 232282
    num_examples: 200
  - name: test
    num_bytes: 232588
    num_examples: 200
  download_size: 1975228
  dataset_size: 2325961
configs:
- config_name: m3-4
  data_files:
  - split: train
    path: m3-4/train-*
  - split: validation
    path: m3-4/validation-*
  - split: test
    path: m3-4/test-*
- config_name: m4-6
  data_files:
  - split: train
    path: m4-6/train-*
  - split: validation
    path: m4-6/validation-*
  - split: test
    path: m4-6/test-*
---
