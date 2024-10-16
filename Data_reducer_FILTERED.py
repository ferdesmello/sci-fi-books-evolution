import pandas as pd
import re

#----------------------------------------------------------------------------------
df_shelf = pd.read_csv('./Data/sci-fi_books_SHELF.csv', sep = ';', encoding='utf-8-sig')
df_lists = pd.read_csv('./Data/sci-fi_books_LISTS.csv', sep = ';', encoding='utf-8-sig')

frames = [df_shelf, df_lists]
df = pd.concat(frames, ignore_index=True)
df.to_csv('./Data/sci-fi_books_BRUTE.csv', index=False, sep=';', encoding='utf-8-sig')

#df = df_shelf
print("\nBRUTE Dataframe")
print(df.info())

#----------------------------------------------------------------------------------
# Filtering out books without both synopses and reviews, books without publishing year, and without many ratings

# Cleaning null synopsis and review fields
#-------------------------------------------
def clean_synopsis(row):
    if pd.isna(row):
        return "No synopsis available"
    else:
        return row
    
#-------------------------------------------
def clean_review(row):
    if pd.isna(row):
        return "No review available"
    else:
        return row
   
#-------------------------------------------
# Apply the function to update the 'bracket_content' column
df['synopsis'] = df['synopsis'].apply(clean_synopsis)
df['review'] = df['review'].apply(clean_review)

#-------------------------------------------
# Minimum character length for synopses or reviews
N_c = 100

# Filter out rows where the length of synopses or reviews is shorter than N_c characters
synopsis_mask = df['synopsis'].str.len().fillna(0) >= N_c
review_mask = df['review'].str.len().fillna(0) >= N_c

df = df[synopsis_mask | review_mask]

#-----------------------------------------
# Dropping books without publishing year
df = df.dropna(axis=0, subset=['year'])

#-----------------------------------------
# Dropping books with fewer than N_r ratings

# Minimum number of ratings
N_r = 10

# Filter out rows where the number of ratings is less than N_r
ratings_mask = df['ratings'] >= N_r
df = df[ratings_mask]

#----------------------------------------------------------------------------------
# Cleaning URLs
df['url'] = df['url'].str.strip()

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

# Deletes open and closed parentheses
df.loc[:, "title"] = df["title"].str.replace(r' \(.*\)', '', regex=True)

# Deletes special case of digit error with only open parenthesis
df.loc[:, "title"] = df["title"].str.replace(r' \(.*', '', regex=True)

# Including series value in the case of square brackes in title (2 cases)
#-------------------------------------------
def update_series(row):
    # Check for existing content
    existing_value = row['series']
    
    # Search for the pattern and capture only the content inside the square brackets
    match = re.search(r'\[(.*?)\]', row['title']) # Captures content inside square brackets
    
    # Update only if a match is found and existing_value is None (or any placeholder you use)
    if match and pd.isna(existing_value):
        #print(match.group(1))
        return match.group(1) # Return only the content inside the brackets
    else:
        return existing_value # Keep the existing value unchanged
    
#-------------------------------------------
# Apply the function to update the 'bracket_content' column
df['series'] = df.apply(update_series, axis=1)

# Deletes special case of square brackets
df.loc[:, "title"] = df["title"].str.replace(r' \[.*', '', regex=True)

#----------------------------------------------------------------------------------
# Excluding colections of books (eg. many books together in one volume)
# Some collections have just "/" separating titles, but some titles use "/" right and I want to keep those
# (must be done after excluding parentheses, as some series have " / " in the parentheses)

def filter_bar(title):
    # Exceptions for exclusion (some books use "/" properly)
    # Use a set for faster lookups
    exceptions = {"11/22/63", 
                  "The After/Life",
                  "The Mighty Thor, Vol. 3: The Asgard/Shi'ar War",
                  "The 7 1/2 Deaths of Evelyn Hardcastle"} 

    # Normalize title by stripping extra spaces
    title = title.strip()

    # Check if the title is in exceptions
    if title in exceptions:
        #print(f"Included as exception: {title}")
        return True

    # Check for unwanted slashes
    if re.search("/", title):
        #print(f"Excluded due to slash: {title}")
        return False
    
    # Check for unwanted anti-slashes
    if re.search(r'\\', title):
        #print(f"Excluded due to slash: {title}")
        return False

    # Include titles without slashes
    return True

# Apply the filter to exclude undesired titles
mask = df['title'].apply(filter_bar)
df = df[mask]

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
# Fixing the many spaces in some author names

# Function to normalize whitespace in the author names
def clean_whitespace(text):
    if isinstance(text, str):
        # Split the string into words and join with a single space
        author_clean = ' '.join(text.split())
        return author_clean # Returns the name cleaned
    return text # Or returns the orignal value if not a string

# Apply the cleaning function to the 'author' column
df['author'] = df['author'].apply(clean_whitespace)

#----------------------------------------------------------------------------------
# Excluding duplicates (some duplicates differ just by capitalization or apostrophe type: ’ or ')

# Function to normalize titles by replacing typographic quotes with standard ones
def normalize_apostrophe(title):
    # Replace right single quotation mark with a straight apostrophe
    normalized_title = title.replace('’', "'")
    return normalized_title

# Apply the normalization function to the title column
df['title'] = df['title'].apply(normalize_apostrophe)

#-------------------------------------------
# Deleting spaces using strip
df['title'] = df['title'].apply(lambda x: x.strip())
df['author'] = df['author'].apply(lambda x: x.strip())

# Create temporary lowercase columns for comparison
df['title_lower'] = df['title'].str.lower()
df['author_lower'] = df['author'].str.lower()

# Drop duplicates based on the lowercase columns
df = df.drop_duplicates(subset=['title_lower', 'author_lower'], keep='first')
df = df.drop_duplicates(subset=['url'], keep='first')

# Drop the temporary lowercase columns
df = df.drop(columns=['title_lower', 'author_lower'])

# Special case of author names
pattern_1 = "Anne; Lackey McCaffrey, Mercedes Lackey"
pattern_2 = "Anne; Lackey McCaffrey"
names = "Anne McCaffrey, Mercedes Lackey"
df['author'] = df['author'].replace(pattern_1, names)
df['author'] = df['author'].replace(pattern_2, names)

#----------------------------------------------------------------------------------
# Deleting some left over duplicates, unwanted non-fiction, and collections

def delete_books(row):
    # Titles to be deleted
    titles_to_del = ["Feersum Endjinn",
                     "Frankenstein: The 1818 Text",
                     "Hard to Be a God",
                     "R.U.R.: Rossum's Universal Robots",
                     "Rama Revealed: The Ultimate Encounter",
                     "Simulacron 3",
                     "Simulacron Three",
                     "Fiction 2000: Cyberpunk and the Future of Narrative",
                     "GURPS Reign of Steel: The War Is Over, The Robots Won",
                     "The Third Time Travel MEGAPACK ®: 18 Classic Trips Through Time",
                     "The Zombie Survival Guide: Complete Protection from the Living Dead",
                     "Mickey 7",
                     "From the Earth to the Moon and 'Round the Moon",
                     "Shards of Honour",
                     "The Men From P.I.G. And R.O.B.O.T.",
                     "H.G. Wells: Seven Novels",
                     "A Handful of Darkness",
                     "Time Patrolman",
                     "Apocalypses",
                     "The Island of Doctor Moreau",
                     "Future Bright, Future Grimm: Transhumanist Tales for Mother Nature's Offspring",
                     "Diaspora: The dark, post-apocalyptic thriller perfect for fans of BLACK MIRROR and Philip K. Dick",
                     "Fahrenheit 451; The Illustrated Man; Dandelion Wine; The Golden Apples of the Sun; The Martian Chronicles",
                     r"Vorkosigan's Game: The Vor Game \ Borders of Infinity",
                     "Miles Errant",
                     "Miles in Love",
                     "Miles, Mutants, and Microbes",
                     "Miles, Mystery & Mayhem",
                     "Miles, Mystery, and Mayhem",
                     "A City in the North: reconsidered",
                     "Second Stage Lesmam",
                     "This Star Shall Abide: aka Heritage of the Star",
                     "Fairyland",
                     "Gunner Cade & Takeoff",
                     "Null States: Book Two of the Centenal Cycle",
                     "Omnitopia: Dawn",
                     "Time and Mr. Bass: A Mushroom Planet Book",
                     "Wool Omnibus",
                     "Alliance Space",
                     "A World Divided",
                     "The Forbidden Circle",
                     "To Save a World",]
    
    # Authors to be deleted
    authors_to_del = ["Iain M. Banks",
                      "Mary Wollstonecraft Shelley",
                      "Arkadi Strugatski",
                      "Josef/Karel Capek",
                      "Arthur C. Clarke",
                      "Daniel F. Galouye",
                      "George Edgar Slusser",
                      "David L. Pulver",
                      "Philip K. Dick",
                      "Max Brooks",
                      "Edward Ashton",
                      "Jules Verne",
                      "Lois McMaster Bujold",
                      "Harry Harrison",
                      "H.G. Wells",
                      "Philip K. Dick",
                      "Poul Anderson",
                      "R.A. Lafferty",
                      "D.J. MacLennan",
                      "Greg Egan",
                      "Ray Bradbury",
                      "Lois McMaster Bujold",
                      "Marta Randall",
                      'E.E. "Doc" Smith',
                      "Star	Sylvia Engdahl",
                      "Paul McAuley",
                      "Cyril Judd",
                      "Malka Ann Older",
                      "Diane Duane",
                      "Eleanor Cameron",
                      "Hugh Howey",
                      "C.J. Cherryh",
                      "Marion Zimmer Bradley"]
    
    # Extract title and author from the row
    title = row['title']
    author = row['author']

    # Check if both title and author match those in the deletion lists
    if (title in titles_to_del) & (author in authors_to_del):
        #print(f"Excluded: {title} by {author}")
        return False
    else:
        return True

# Apply the filter function to each row using axis=1 to access row data
mask = df.apply(delete_books, axis=1)
df = df[mask]

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
#df = df.loc[indices_to_keep_1] # Uncomment this line to exclude books with 'fantasy' as first genre

#-----------------------------------------
# Define unwanted and required genres
unwanted_genres = ['Graphic Novels', 
                   'Comics', 
                   'Graphic Novels Comics', 
                   'Comic Book', 
                   'Manga', 
                   'Short Stories', 
                   'Anthologies', 
                   'Nonfiction', 
                   'Art', 
                   'Reference',
                   'Literary Criticism', 
                   'Essays', 
                   'Criticism',
                   "Role Playing Games",
                   'High Fantasy',
                   'Epic Fantasy',
                   'Magic',
                   'Angels']
unwanted_genres = [genre.lower() for genre in unwanted_genres]

#required_genres = ['Short Stories', 'Anthologies']
#required_genres = [genre.lower() for genre in required_genres]

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
    # 2. The list must contain any of the required genres (if applicable)
    # 3. The list must contain the required genre
    has_unwanted_genres = any(genre in unwanted_genres for genre in processed_genres)
    #has_required_genres = any(genre in required_genres for genre in processed_genres)
    has_required_genre = required_genre in processed_genres

    # Add the index if the row does not have any unwanted genres and has the required genre
    #if not has_unwanted_genres and (has_required_genre & has_required_genres):
    if not has_unwanted_genres and has_required_genre:
        indices_to_keep_2.append(index)

# Create the filtered DataFrame using the indices of rows to keep
df_filtered = df.loc[indices_to_keep_2]

df_filtered = df_filtered.sort_values(by=['decade', 'year', 'author', 'title'], axis=0, ascending=True)

#----------------------------------------------------------------------------------
# Save the filtered DataFrame back to a CSV

print("\nFILTERED Dataframe")
print(df_filtered.info())

df_filtered.to_csv('./Data/sci-fi_books_FILTERED.csv', index=False, sep=';', encoding='utf-8-sig')
print(f"\nData saved to ./Data/sci-fi_books_FILTERED.csv")