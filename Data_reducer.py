import pandas as pd

#----------------------------------------------------------------------------------
df_shelf = pd.read_csv('sci-fi_books_shelf.csv', sep = ';')
df_lists = pd.read_csv('sci-fi_books_lists.csv', sep = ';')

frames = [df_shelf, df_lists]
#df = pd.concat(frames, ignore_index=True)
#df.to_csv('sci-fi_books_BRUTE.csv', index=False, sep=';')

df = df_shelf
print(df.info())

#----------------------------------------------------------------------------------
# Excluding series of books (eg. trilogies together in one volume)
# (must be before excluding parentheses, as this info is in the parentheses)

# Regex pattern for exclusion ("#1-3" for books of a series together)
pattern = r"#1-\d"
# Exclude rows where the 'title' column matches the pattern
df = df[~df['title'].str.contains(pattern, regex=True)]

"""#----------------------------------------------------------------------------------
# Column indicating if the book is part of a series or not

# Regex pattern to identify ("#1)")
pattern = r"#\d{1,2}\)"

# Create a new column indicating if the book is part of a series
df['series'] = (df['title'].str.contains(pattern, regex=True)
                .apply(lambda x: 'yes' if x else 'no'))"""

#----------------------------------------------------------------------------------
# Deleting parentheses from titles
# (must be after excluding series together in a volume, as some info is in the parentheses)
df.loc[:, "title"] = df["title"].str.replace(r' \(.*\)', '', regex=True)

#----------------------------------------------------------------------------------
# Excluding colections of books (eg. many books together in one volume)
# (must be after excluding parentheses, as some series have " / " in the parentheses)

# Regex pattern for exclusion (" / " for many books in one volume)
pattern = r" / "
# Exclude rows where the 'title' column matches the pattern
mask = df['title'].str.contains(pattern, regex=True)
df = df[~mask]

#----------------------------------------------------------------------------------
# Coding the series field
df['series'] = df['series'].notna().map({True: 'yes', False: 'no'})

#----------------------------------------------------------------------------------
# Cleaning the pages field

# Exclude the pattern from the pages column
#df['pages'] = df['pages'].str.replace(' pages', '')
#df = df.astype({'page': 'int64'})

# Update the existing column to keep only the number part before the space
df['pages'] = df['pages'].str.extract(r'(\d+)', expand=False)

# Optionally, convert extracted values to integers (if all values are numeric)
df['pages'] = df['pages'].fillna(0).astype(int)

#----------------------------------------------------------------------------------
# Excluding duplicates

# Remove duplicates based on specific columns
df = df.drop_duplicates(subset=['title', 'author'], keep = 'first')

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

# Example: Filter for books from the 1960s and order by ratings
"""books_1960s = grouped[grouped['decade'] == 1960]
print("\nBooks from the 1960s ordered by ratings:")
print(books_1960s)"""

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
df_filtered = df.loc[indices_to_keep_2]#.reset_index(drop=True)

print(df_filtered.info())

#----------------------------------------------------------------------------------
# Top 200 rating books for every decade

# Group by 'decade', sort by 'ratings' in descending order, and select the top 200 per group
df_top_books = (df_filtered.groupby('decade', group_keys=False)
                .apply((lambda x: x.sort_values('ratings', ascending=False).head(200)), 
                       include_groups=False))

# Decade has been lost in the grupby above, so reintroducing it below
decade_index = []
for index, _ in df_top_books.iterrows():
    decade_index.append(index)
df_top_books['decade'] = df_filtered['decade'].loc[decade_index]
df_top_books = df_top_books.reset_index(drop=True)

# Reordering columns
column_order = ['title', 
                'author', 
                'year',
                'decade', 
                'pages', 
                'rate', 
                'ratings', 
                'series', 
                'genres', 
                'synopsis',
                'review',
                'url']
df_top_books = df_top_books.reindex(columns=column_order)

df_top_books.to_csv('top_sci-fi_books_200.csv', index=False, sep=';')

print(df_top_books.info())
#----------------------------------------------------------------------------------
# Sample of the total for testing

test_books = [
    "Dune",
    "The Hitchhiker’s Guide to the Galaxy",
    "1984",
    "Brave New World",
    "The Forever War",
    "Stranger in a Strange Land",
    "Childhood’s End",
    "The Time Machine",
    "Twenty Thousand Leagues Under the Sea",
    "The Left Hand of Darkness",
    "Contact",
    "Blindness",
    "Annihilation",
    "Solaris",
    "Foundation",
    "Last and First Men",
    "The Sparrow",
    "The Player of Games",
    "Blindsight",
    "The Fountains of Paradise",
    "Sirius"
]
test_books_mask = df_filtered['title'].isin(test_books)
df_test_books = df_filtered[test_books_mask]

df_test_books = df_test_books.reindex(columns=column_order)
df_test_books = df_test_books.sort_values(by=['ratings'], axis=0, ascending=False)

df_test_books.to_csv('top_books_TEST.csv', index=False, sep=';')

#----------------------------------------------------------------------------------
# Save the filtered DataFrame back to a CSV

df_filtered.to_csv('sci-fi_books_FILTERED.csv', index=False, sep=';')
print(f"\nData saved to sci-fi_books_FILTERED.csv")