import pandas as pd

#----------------------------------------------------------------------------------
# Load files
df_200 = pd.read_csv("./Data/top_sci-fi_books_200_PER_DECADE.csv", sep=';', encoding="utf-8-sig")
df_AI_answers = pd.read_csv('./Data/AI_answers_to_sci-fi_books.csv', sep=';', encoding="utf-8-sig")

print(df_200.info())
print(df_AI_answers.info())

#----------------------------------------------------------------------------------
"""# Merge df_A with df_B on the common column 'id'
df_merged = pd.merge(df_AI_answers, df_200[['url', 'synopsis', 'review']], on='url', how='inner', suffixes=('_A', '_B'))
#print(df_merged.info())

# Replace data in df_A columns with values from df_B (replace col_X_A with col_X_B)
df_merged['synopsis_A'] = df_merged['synopsis_B']
df_merged['review_A'] = df_merged['review_B']
#print(df_merged.info())

# Drop the extra column from df_B (col_X_B)
df_merged = df_merged.drop(columns=['synopsis_B', 'review_B'])
#print(df_merged.info())

# Rename the column back to its original name if needed
df_merged = df_merged.rename(columns={'synopsis_A': 'synopsis', 'review_A': 'review'})
#print(df_merged.info())

df_merged = df_merged.drop_duplicates(subset=['url'], keep='first')
#print(df_merged.info())

# Remake dataframe
df_AI_answers = df_merged"""

#----------------------------------------------------------------------------------

# Merge df_A with df_B on the common column 'id'
df_merged = pd.merge(df_AI_answers, df_200[['url', 'rate', 'ratings']], on='url', how='inner')

df_AI_answers = df_merged

column_order = ['title', 
                'author', 
                'year',
                'paragraph',
                '1 soft hard',
                'justifying soft hard',
                '2 time',
                'justifying time',
                '3 tone',
                'justifying tone',
                '4 setting',
                'justifying setting',
                '5 on Earth',
                'justifying on Earth',
                '6 post apocalyptic',
                'justifying post apocalyptic',
                '7 aliens',
                'justifying aliens',
                '8 aliens are',
                'justifying aliens are',
                '9 robots and AI',
                'justifying robots and AI',
                '10 robots and AI are',
                'justifying robots and AI are',
                '11 tech and science',
                'justifying tech and science',
                '12 protagonist',
                'justifying protagonist',
                '13 social issues',
                'justifying social issues',
                '14 enviromental',
                'justifying enviromental',
                'complete answer',
                'decade',
                'rate', 
                'ratings', 
                'series', 
                'genres', 
                'synopsis',
                'review',
                'url']

df_AI_answers = df_AI_answers.reindex(columns=column_order)

# Retyping
df_AI_answers['year'] = df_AI_answers['year'].astype(int)
df_AI_answers['decade'] = df_AI_answers['decade'].astype(int)
df_AI_answers['rate'] = df_AI_answers['rate'].astype(float)
df_AI_answers['ratings'] = df_AI_answers['ratings'].astype(int)

print(df_AI_answers.info())

#----------------------------------------------------------------------------------
df_AI_answers.to_csv('./Data/AI_ANSWERS_TO_sci-fi_books.csv', index=False, sep=';', encoding='utf-8-sig')
print(f"Data saved to ./Data/AI_ANSWERS_TO_sci-fi_books.csv")