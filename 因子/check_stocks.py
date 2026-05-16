import pandas as pd
df = pd.read_parquet(r"C:\因子数据\_stock_list.parquet")
print(f"Stock list: {len(df)} rows, cols: {list(df.columns)}")
print(df.head(3))
print(f"Types:\n{df['type'].value_counts()}")
stocks = df[df['type'] == '1']
print(f"A-share stocks (type=1): {len(stocks)}")
print(stocks.head(5))
