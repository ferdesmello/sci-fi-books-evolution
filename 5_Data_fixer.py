import pandas as pd

#----------------------------------------------------------------------------------
df_top = pd.read_csv('./Data/sci-fi_books_TOP.csv', sep = ';', encoding="utf-8-sig")
df_AI = pd.read_csv('./Data/sci-fi_books_AI_ANSWERS.csv', sep = ';', encoding="utf-8-sig")
df_AI_gender = pd.read_csv('./Data/sci-fi_books_AI_ANSWERS_GENDER.csv', sep = ';', encoding="utf-8-sig")

#----------------------------------------------------------------------------------
# Cleaning the sci-fi_books_AI_ANSWERS.csv file, so it has only sci-fi_books_TOP.csv books.

column_names = df_AI.columns
df_cleaned = pd.DataFrame(columns = column_names)
books = set(df_top['url'])
counter = 0

for index, row in df_AI.iterrows():
    if row['url'] in books:
        df_cleaned = pd.concat([df_cleaned, row.to_frame().T], ignore_index=True)
    else:
        counter += 1
        continue

print(f"{counter} book(s) have been excluded from sci-fi_books_FILTERED.csv.")

#----------------------------------------------------------------------------------
# Cleaning the sci-fi_books_AI_ANSWERS_GENDER.csv file, so it has only sci-fi_books_TOP.csv authors.

column_names_gender = df_AI_gender.columns
df_cleaned_gender = pd.DataFrame(columns = column_names_gender)
names = set(df_top['author'])
counter_gender = 0

for index_gender, row_gender in df_AI_gender.iterrows():
    if row_gender['author'] in names:
        df_cleaned_gender = pd.concat([df_cleaned_gender, row_gender.to_frame().T], ignore_index=True)
    else:
        counter_gender += 1
        continue

print(f"{counter_gender} name(s) have been excluded from sci-fi_books_AI_ANSWERS_GENDER.csv.")

#----------------------------------------------------------------------------------
df_cleaned.to_csv('./Data/sci-fi_books_AI_ANSWERS.csv', index=False, sep=';', encoding='utf-8-sig')
df_cleaned_gender.to_csv('./Data/sci-fi_books_AI_ANSWERS_GENDER.csv', index=False, sep=';', encoding='utf-8-sig')