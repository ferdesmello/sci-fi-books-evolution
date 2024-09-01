from bs4 import BeautifulSoup
import pandas as pd
import os

#----------------------------------------------------
def extract_books_from_file(file_path):
    # Open and read the saved HTML file
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Parse the HTML content with BeautifulSoup
    soup = BeautifulSoup(content, 'html.parser')
    
    books = []
    # Find all book elements on the page
    book_elements = soup.find_all('div', class_='elementList')
    
    for book in book_elements:
        title_elem = book.find('a', class_='bookTitle')
        author_elem = book.find('a', class_='authorName')
        span_element = book.find('span', class_='greyText smallText')  # Find within each book element

        # Extract and clean the book details
        if title_elem and author_elem and span_element:
            title = title_elem.text.strip()
            author = author_elem.text.strip()
            text_split = span_element.text.split('—')
            
            # Handle potential variations in the text structure
            avg_rating = text_split[0].split()[-1] if len(text_split) > 0 else 'N/A'
            ratings = text_split[1].split()[0].replace(',', '') if len(text_split) > 1 else 'N/A'
            year = text_split[2].split()[-1] if len(text_split) > 2 else 'N/A'

            books.append({
                'title': title,
                'author': author,
                'year': int(year) if year.isdigit() else 'N/A',
                'avg_rating': float(avg_rating) if avg_rating.replace('.', '', 1).isdigit() else 'N/A',
                'ratings': int(ratings) if ratings.isdigit() else 'N/A',
                'url': title_elem['href'],
            })
    
    return books

#----------------------------------------------------
# Directory containing the saved HTML files
directory = './saved_pages'  # Update this path to where your HTML files are stored

all_books = []

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.html'):
        file_path = os.path.join(directory, filename)
        print(f"Processing file: {file_path}")
        books = extract_books_from_file(file_path)
        all_books.extend(books)

#----------------------------------------------------
# Convert the list of books to a DataFrame and save it to a CSV
df = pd.DataFrame(all_books)
df.to_csv('sci-fi_books_extracted.csv', index=False, sep=';')

print(f"Extracted data from {len(all_books)} books across all pages.")
