import pandas as pd

#----------------------------------------------------------------------------------
df_shelf = pd.read_csv('./Data/sci-fi_books_SHELF.csv', sep = ';')
df_lists = pd.read_csv('./Data/sci-fi_books_LISTS.csv', sep = ';')

frames = [df_shelf, df_lists]
df = pd.concat(frames, ignore_index=True)
df.to_csv('./Data/sci-fi_books_BRUTE.csv', index=False, sep=';')

#df = df_shelf
print("\nBRUTE Dataframe")
print(df.info())

#----------------------------------------------------------------------------------
# Excluding series of books (eg. trilogies together in one volume)
# (must be done before excluding parentheses, as this info is in the parentheses)

# Regex pattern for exclusion ("#1-3" for books of a series together)
pattern = r"#1-\d"

# Exclude rows where the 'title' column matches the pattern
mask_titles = df['title'].str.contains(pattern, regex=True, na=False)
df = df[~mask_titles]

# Regex pattern for exclusion ("#1-3" for books of a series together)
pattern = r"#\d-\d"

# Exclude rows where the 'series' column matches the pattern
mask_series = df['series'].str.contains(pattern, regex=True, na=False)
df = df[~mask_series]

#----------------------------------------------------------------------------------
# Deleting parentheses from titles
# (must be done after excluding series together in a volume, as some info is in the parentheses)

df.loc[:, "title"] = df["title"].str.replace(r' \(.*\)', '', regex=True)

#----------------------------------------------------------------------------------
# Excluding colections of books (eg. many books together in one volume)
# (must be done after excluding parentheses, as some series have " / " in the parentheses)

# Regex pattern for exclusion (" / " for many books in one volume)
pattern = r" / "

# Exclude rows where the 'title' column matches the pattern
mask = df['title'].str.contains(pattern, regex=True, na=False)
df = df[~mask]

#----------------------------------------------------------------------------------
# Coding the series field
df['series'] = df['series'].notna().map({True: 'yes', False: 'no'})

#----------------------------------------------------------------------------------
# Cleaning the pages field

# Update the existing column to keep only the number part before the space
df['pages'] = df['pages'].str.extract(r'(\d+)', expand=False)

# Optionally, convert extracted values to integers (if all values are numeric)
df['pages'] = df['pages'].fillna(0).astype(int)

#----------------------------------------------------------------------------------
# Excluding duplicates (some duplicates differ just by capitalization of titles)

# Create temporary lowercase columns for comparison
df['title_lower'] = df['title'].str.lower()
df['author_lower'] = df['author'].str.lower()

# Drop duplicates based on the lowercase columns
df = df.drop_duplicates(subset=['title_lower', 'author_lower'], keep='first')

# Drop the temporary lowercase columns
df = df.drop(columns=['title_lower', 'author_lower'])

#----------------------------------------------------------------------------------
# Filtering out synopses too short and books without publishing year

# Minimum character length for the synopsis
N = 100

# dropping nulls NaN so it can compare lengths
df = df.dropna(axis=0, subset=['synopsis'])

# Filter out rows where the length of the synopsis is shorter than N characters
synopsis_mask = df['synopsis'].str.len().fillna(0) >= N
df = df[synopsis_mask]

df = df.dropna(axis=0, subset=['year'])

#----------------------------------------------------------------------------------
# Converting the genres' column to actual lists
for index, row in df.iterrows():
    genres = row['genres']
    # Convert the string representation of lists into actual lists
    if isinstance(genres, str):
        df.at[index, 'genres'] = eval(genres)

#----------------------------------------------------------------------------------
# Grouping by decade

# Create a 'decade' column
# Floor division by 10, then multiply by 10 to get the start of the decade
df['decade'] = (df['year'] // 10) * 10 

# Group by decade and sort within each group by 'ratings' in descending order
grouped = df.sort_values(by = ['decade', 'ratings'], ascending = [True, False])

df = df.astype({'year': 'int64', 'decade': 'int64'})

#----------------------------------------------------------------------------------
# Filtering out genres
#-----------------------------------------
# Excluding books with fantasy as first genre

# Initialize an empty list to hold indices of rows to keep
indices_to_keep_1 = []

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    genres = row['genres']
    # Check if the genres list is not empty and if the first genre is not 'fantasy'
    if genres and genres[0].lower() == 'fantasy':
        # Skip this row, i.e., don't add it to the indices_to_keep list
        continue
    # Add the index of the row to keep it
    indices_to_keep_1.append(index)

# Create the filtered DataFrame using the indices of rows to keep
df = df.loc[indices_to_keep_1]

#-----------------------------------------
# Define unwanted and required genres
unwanted_genres = ['Graphic Novels', 
                   'Comics', 
                   'Graphic Novels Comics', 
                   'Comic Book', 
                   'Manga', 
                   'Short Stories', 
                   'Anthologies', 
                   'Collections', 
                   'Nonfiction', 
                   'Art', 
                   'Reference']

unwanted_genres = [genre.lower() for genre in unwanted_genres]

required_genre = 'science fiction'

# Initialize an empty list to keep track of indices of rows to keep
indices_to_keep_2 = []

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    genres_line = row['genres']
    
    # Ensure genres are processed as lowercase and stripped of extra spaces
    processed_genres = [genre.lower().strip() for genre in genres_line]

    # Check if the row should be kept:
    # 1. The list of genres must not contain any unwanted genres
    # 2. The list must contain the required genre
    has_unwanted_genre = any(genre in unwanted_genres for genre in processed_genres)
    has_required_genre = required_genre in processed_genres

    # Add the index if the row does not have any unwanted genres and has the required genre
    if not has_unwanted_genre and has_required_genre:
        indices_to_keep_2.append(index)

# Create the filtered DataFrame using the indices of rows to keep
df_filtered = df.loc[indices_to_keep_2]

#----------------------------------------------------------------------------------
# Save the filtered DataFrame back to a CSV

print("\nFILTERED Dataframe")
print(df_filtered.info())

df_filtered.to_csv('./Data/sci-fi_books_FILTERED.csv', index=False, sep=';')
print(f"\nData saved to ./Data/sci-fi_books_FILTERED.csv")