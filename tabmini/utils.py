

def find_almost_constant_columns(df, threshold, working_directory, dt_name, num_records):
    """
    Find columns in df where the most frequent value occupies >= threshold% of rows
    Default threshold=0.9 (ie 90%)
    """

    columns_list = []
    n_rows = len(df)
    
    for col in df.columns:
        # Tính tần suất mỗi giá trị (kể cả NaN nếu muốn)
        value_counts = df[col].value_counts(dropna=False)
        
        # Tần suất lớn nhất
        max_count = value_counts.max()
        
        # Kiểm tra tỉ lệ max_count so với tổng số dòng
        if max_count / n_rows >= threshold:
            columns_list.append(col)
    
    if len(columns_list) > 0:
      filename= working_directory / f"{dt_name}_{num_records}_columns_deleted.csv"
      with open(filename, "w", encoding="utf-8") as f:
          for col in columns_list:
              f.write(col + "\n")

    return df.drop(columns=columns_list)