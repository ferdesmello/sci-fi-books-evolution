import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import random
import logging
import re

#----------------------------------------------------------------------------------
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#----------------------------------------------------------------------------------
# Function to make session as a browser
def get_session():
    session = requests.Session()
    retries = Retry(total=10, backoff_factor=0.2, status_forcelist=[413, 429, 500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    return session

#----------------------------------------------------------------------------------
# Function to scrape data from the local shelf HTML files
def scrape_shelf_from_html(file_path):
    books = []
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        content = file.read()
        soup = BeautifulSoup(content, 'html.parser')

        #-----------------------------------------
        # Approach 1: Extract from shelves using <div> elements with class "elementList"
        for book in soup.find_all('div', class_='elementList'):
            title_elem = book.find('a', class_='bookTitle')
            author_elem = book.find('a', class_='authorName')

            if title_elem and author_elem:
                books.append({
                    'title': title_elem.text.strip(),
                    'author': author_elem.text.strip(),
                    'url': title_elem['href']
                })

        #-----------------------------------------
        # Approach 2: Extract from lists using <tr> elements with itemtype="http://schema.org/Book"
        for book in soup.find_all('tr', itemtype='http://schema.org/Book'):
            title_elem = book.find('a', class_='bookTitle')
            author_elem = book.find('a', class_='authorName')

            if title_elem and author_elem:
                books.append({
                    'title': title_elem.find('span', itemprop='name').text.strip(),
                    'author': author_elem.find('span', itemprop='name').text.strip(),
                    'url': title_elem['href']
                })

    return books

#----------------------------------------------------------------------------------
# Function to scrape data from book pages
def scrape_book_page(session, url):
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        #-----------------------------------------
        # Extract series
        series_tag = soup.find('h3', class_='Text Text__title3 Text__italic Text__regular Text__subdued')
        series = series_tag.text.strip() if series_tag else None

        #-----------------------------------------
        # Extract pages
        pages_tag = soup.find('p', attrs={'data-testid': 'pagesFormat'})
        pages = pages_tag.text.strip() if pages_tag else None

        #-----------------------------------------
        # Extract year
        year = None
        publication_info = soup.find('p', {'data-testid': 'publicationInfo'})
        if publication_info:
            year_text = publication_info.get_text()
            match = re.search(r'(\d{4})', year_text)
            if match:
                year = match.group(1)

        #-----------------------------------------
        # Extract genres
        genres = []
        genre_elem = soup.find('div', {'data-testid': 'genresList'})
        if genre_elem:
            genres = [a.text.strip() for a in genre_elem.find_all('a')]

        #-----------------------------------------
        # Extract synopsis
        synopsis = "No synopsis available"
        description_elem = soup.find('div', {'data-testid': 'description'})
        if description_elem:
            synopsis = description_elem.text.strip()

        #-----------------------------------------
        # Extract average rate
        rate = 0
        rating_div = soup.find('div', class_='RatingStatistics__rating')
        if rating_div:
            rate_text = rating_div.text.strip()
            rate = float(rate_text)

        #-----------------------------------------
        # Extract number of ratings
        ratings = 0
        ratings_elem = soup.find('span', {'data-testid': 'ratingsCount'})
        if ratings_elem:
            ratings_text = ratings_elem.text.strip().split()[0].replace(',', '')
            ratings = int(ratings_text)

        #-----------------------------------------
        # Find all review sections
        review_sections = soup.find_all('section', class_='ReviewText')

        # Extract first and second reviews if they exist
        review_1 = review_sections[0].find('span', class_='Formatted').text.strip() if len(review_sections) > 0 else "No review available"
        review_2 = review_sections[1].find('span', class_='Formatted').text.strip() if len(review_sections) > 1 else "No review available"

        # Get the longer review
        if len(review_2) > len(review_1):
            review = review_2
        else:
            review = review_1

        # Display the extracted reviews
        #print(f"\nFirst review: {review_1}")
        #print(f"\nSecond review: {review_2}")
        #print(f"\nChosen review: {review}")

        #-----------------------------------------
        # Book data extracted
        book_data = {
            'series': series,
            'pages': pages,
            'year': year,
            'rate': rate,
            'ratings': ratings,
            'genres': genres,
            'synopsis': synopsis,
            'review': review,
        }

        logging.info(f"Successfully scraped book:\n  {url}")
        logging.debug(f"Book data: {book_data}")

        return book_data

    except Exception as e:
        logging.error(f"Error scraping\n!!!{url}: {e}")
        return None

#----------------------------------------------------------------------------------
# Main scraping function
def scrape_goodreads_books_from_files(folder_path):
    session = get_session()
    all_books = []

    # Read all HTML files from the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.html'):
            file_path = os.path.join(folder_path, file_name)
            logging.info(f"Scraping file:-----------------\n{file_path}")
            books = scrape_shelf_from_html(file_path)
            for book in books:
                book_data = scrape_book_page(session, book['url'])
                if book_data:
                    book.update(book_data)
                    all_books.append(book)
                    #print(book)
                else:
                    logging.warning(f"Failed to scrape book:\n!!!{book['url']}")

            # Implement a random delay between 5 to 15 seconds after every page (not book)
            time.sleep(random.uniform(5, 15))

    return all_books

#----------------------------------------------------------------------------------
# Main execution function
def main():
    folder_path = './Saved_pages'
    books = scrape_goodreads_books_from_files(folder_path)
    df = pd.DataFrame(books)

    #--------------------------------------------
    # Chose the right columns and their order
    column_order = ['title', 
                    'author', 
                    'year', 
                    'pages', 
                    'rate', 
                    'ratings', 
                    'series', 
                    'genres', 
                    'synopsis',
                    'review',
                    'url']
    
    df = df.reindex(columns=column_order)

    #--------------------------------------------
    # Save the final DataFrame to a CSV file
    df.to_csv('./Data/sci-fi_books_SHELF.csv', index=False, sep=';', encoding='utf-8-sig')

    logging.info(f"Scraped {len(books)} books.\nData saved to ./Data/sci-fi_books_SHELF.csv")

#----------------------------------------------------------------------------------
# Execution
if __name__ == "__main__":
    main()