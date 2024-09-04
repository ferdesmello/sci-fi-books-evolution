import pandas as pd

#----------------------------------------------------------------------------------
df_shelf = pd.read_csv('sci-fi_books_shelf.csv', sep = ';')
#df_lists = pd.read_csv('sci-fi_books_lists.csv', sep = ';')

print(df_shelf.info())
#print(df_lists.info())

#frames = [df_shelf, df_lists]
#df = pd.concat(frames)

df = df_shelf
#df = df_lists
#print(df.info())

#----------------------------------------------------------------------------------
# Deleting parentheses from titles
df["title"] = df["title"].str.replace(r' \(.*\)', '', regex = True)

# Sorting by year
df.sort_values(by = ['year'], axis = 0, ascending = True, inplace = True)

#----------------------------------------------------------------------------------
# Converting the genres' column to actual lists
for index, row in df.iterrows():
    genres = row['genres']
    # Convert the string representation of lists into actual lists
    if isinstance(genres, str):
        df.at[index, 'genres'] = eval(genres)

#----------------------------------------------------------------------------------
# Excluding duplicates

# Remove duplicates based on all columns (default behavior)
#df = df.drop_duplicates()

# Remove duplicates based on specific columns
df.drop_duplicates(subset=['title', 'author'], keep = 'first', inplace = True)

#----------------------------------------------------------------------------------
# Grouping by decade

# Create a 'decade' column
# Floor division by 10, then multiply by 10 to get the start of the decade
df['decade'] = (df['year'] // 10) * 10 

# Group by decade and sort within each group by 'ratings' in descending order
grouped = df.sort_values(by = ['decade', 'ratings'], ascending = [True, False])

# Example: Filter for books from the 1960s and order by ratings
"""books_1960s = grouped[grouped['decade'] == 1960]
print("\nBooks from the 1960s ordered by ratings:")
print(books_1960s)"""

#----------------------------------------------------------------------------------
# Filtering out synopses too short

# Minimum character length for the synopsis
N = 100
# dropping nulls NaN so it can compare lengths
df.dropna(axis=0, subset=['synopsis'], inplace=True)
# Filter out rows where the length of the synopsis is shorter than N characters
synopsis_mask = df['synopsis'].str.len().fillna(0) >= N
df = df[synopsis_mask]

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
# Excluding other genres
unwanted_genres = ['Graphic Novels',
                   'Comics',
                   'Graphic Novels Comics', 
                   'Comic Book', 
                   'Manga',
                   'Short Stories']
unwanted_genres = list(map(str.lower, unwanted_genres))
required_genre = 'Science Fiction'.lower()

# Initialize an empty list to keep track of indices of rows to keep
indices_to_keep_2 = []

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    genres = row['genres']
    # Check if the row should be kept:
    # 1. The list of genres must not contain any unwanted genres
    # 2. The list must contain the required genre
    has_unwanted_genre = any(genre.lower() in unwanted_genres for genre in genres)
    has_required_genre = required_genre.lower() in [genre.lower() for genre in genres]

    # Add the index if the row does not have any unwanted genres and has the required genre
    if not has_unwanted_genre and has_required_genre:
        indices_to_keep_2.append(index)

# Create the filtered DataFrame using the indices of rows to keep
df_filtered = df.loc[indices_to_keep_2]

print(df_filtered.info())
#print(df_filtered.head())

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
    "Consider Phlebas",
    "The Sparrow",
    "The Player of Games",
    "Blindsight",
    "The Fountains of Paradise",
]

test_books_mask = df_filtered['title'].isin(test_books)
df_top_books = df_filtered[test_books_mask]

column_order = ['title', 'author', 'year', 'ratings', 'synopsis']
df_top_books = df_top_books.reindex(columns=column_order)

df_top_books = df_top_books.sort_values(by=['ratings'], axis=0, ascending=False)

df_top_books.to_csv('sci-fi_top_books.csv', index=False, sep=';')
print(df_top_books)

#----------------------------------------------------------------------------------
# Save the filtered DataFrame back to a CSV

df_filtered.to_csv('sci-fi_books_filtered.csv', index=False, sep=';')
print(f"\nData saved to sci-fi_books_filtered.csv")