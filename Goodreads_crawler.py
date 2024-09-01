import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import json
import os
from requests.exceptions import RequestException
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import random
import logging
import re

#----------------------------------------------------------------------------------
# Set up logging.
logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')

#----------------------------------------------------------------------------------
# Function to load the prior progress in scrapping.
def load_progress():
    try:
        if os.path.exists('scraping_progress.json'):
            with open('scraping_progress.json', 'r') as f:
                return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load progress: {e}")
    
    return {'last_page': 0, 'books': []}

#----------------------------------------------------------------------------------
# Function to save the progress in a json file.
def save_progress(progress):
    try:
        with open('scraping_progress.json', 'w') as f:
            json.dump(progress, f)
        logging.info(f"Progress saved: Last page = {progress['last_page']}")
    except Exception as e:
        logging.error(f"Failed to save progress: {e}")

#----------------------------------------------------------------------------------
# Function to make session as a browser.
def get_session():
    session = requests.Session()
    retries = Retry(total = 5, backoff_factor = 0.1, status_forcelist = [500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries = retries))
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })

    return session

#----------------------------------------------------------------------------------
# Function to scrape data from the shelf pages.
def scrape_shelf(session, page):
    url = f"https://www.goodreads.com/shelf/show/science-fiction?page={page}"
    logging.info(f"Fetching URL: {url}")
    response = session.get(url, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Debug: Print a snippet of the page to check if the content is changing
    logging.debug(f"Page snippet: {soup.text[:500]}")  # Print first 500 characters of the page

    books = []

    for book in soup.find_all('div', class_='elementList'):
        title_elem = book.find('a', class_='bookTitle')
        author_elem = book.find('a', class_='authorName')

        if title_elem and author_elem:
            books.append({
                'title': title_elem.text.strip(),
                'author': author_elem.text.strip(),
                'url': "https://www.goodreads.com" + title_elem['href']
            })

    return books

#----------------------------------------------------------------------------------
# Functin to scrape data from book pages.
def scrape_book_page(session, url):

    try:
        response = session.get(url, timeout = 10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        #------------------------------------
        # Extract year
        year = None
        publication_info = soup.find('p', {'data-testid': 'publicationInfo'})

        if publication_info:
            year_text = publication_info.get_text()
            match = re.search(r'(\d{4})', year_text)

            if match:
                year = match.group(1)

        #------------------------------------
        # Extract genres
        genres = []
        genre_elem = soup.find('div', {'data-testid': 'genresList'})

        if genre_elem:
            genres = [a.text.strip() for a in genre_elem.find_all('a')]

        #------------------------------------
        # Extract synopsis
        synopsis = "No synopsis available"
        description_elem = soup.find('div', {'data-testid': 'description'})

        if description_elem:
            synopsis = description_elem.text.strip()

        #------------------------------------
        # Extract mean rate
        rate = 0
        rating_div = soup.find('div', class_ = 'RatingStatistics__rating')
        if rating_div:
            rate_text = rating_div.text.strip()
            rate = float(rate_text)

        #------------------------------------
        # Extract number of ratings
        ratings = 0
        ratings_elem = soup.find('span', {'data-testid': 'ratingsCount'})

        if ratings_elem:
            ratings_text = ratings_elem.text.strip().split()[0].replace(',', '')
            ratings = int(ratings_text)

        #------------------------------------
        book_data = {
            'year': year,
            #'rate': rate,
            #'ratings': ratings,
            #'genres': genres,
            #'synopsis': synopsis,
        }
        
        logging.info(f"Successfully scraped book: {url}")
        logging.debug(f"Book data: {book_data}")
        
        return book_data
    
    except Exception as e:
        logging.error(f"Error scraping {url}: {e}")

        return None

#----------------------------------------------------------------------------------
# Main scrapinf function.
def scrape_goodreads_scifi(start_page = 1, pages = 100):
  progress = load_progress()
  books = progress.get('books', [])
  last_page = progress.get('last_page', 0)

  # Ensure start_page is updated correctly
  # Option 1: Remove reset logic (commented out)
  # if last_page >= start_page:
  #   start_page = last_page + 1

  session = get_session()
  
  for page in range(start_page, start_page + pages):
    logging.info(f"Scraping page {page}")
    print("\n",page,"\n")

    try:
      page_books = scrape_shelf(session, page)
      if not page_books:
          logging.warning(f"No books found on page {page}. Skipping to next page.")
          continue

      for book in page_books:
        book_data = scrape_book_page(session, book['url'])
        if book_data:
          book.update(book_data)
          books.append(book)
        else:
          logging.warning(f"Failed to scrape book: {book['url']}")

      # Option 2: Update last_page within the loop
      #last_page = page  # Update last_page to the current page

      # Save progress after processing all books on the page
      progress['books'] = books
      progress['last_page'] = last_page
      save_progress(progress)

      # Implement a random delay between 5 to 15 seconds
      time.sleep(random.uniform(5, 15))

    except RequestException as e:
      logging.error(f"Error scraping page {page}: {e}")
      time.sleep(60)  # Wait a minute before retrying
  
  return books

#----------------------------------------------------------------------------------
# Function of main execution.
def main():
    books = scrape_goodreads_scifi(start_page = 1, pages = 3)  # Adjust as needed
    df = pd.DataFrame(books)
    df.to_csv('sci-fi_books.csv', index = False, sep = ';')
    logging.info(f"Scraped {len(books)} books. Data saved to sci-fi_books.csv")

#----------------------------------------------------------------------------------
if __name__ == "__main__":
    main()