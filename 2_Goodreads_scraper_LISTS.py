import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import json
import os
from requests.exceptions import RequestException, Timeout
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import random
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
import re

#----------------------------------------------------------------------------------
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#----------------------------------------------------------------------------------
# Function to load data from the JSON file
def load_progress():
    if os.path.exists('./Data/scraping_progress.json'):
        with open('./Data/scraping_progress.json', 'r') as f:
            return json.load(f)
    return {'urls': {}}

#----------------------------------------------------------------------------------
# Function to dump data in the JSON file
def save_progress(progress):
    with open('./Data/scraping_progress.json', 'w') as f:
        json.dump(progress, f)

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
# Function to retry scraping
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def make_request(session, url, timeout=30):
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        return response
    except Timeout:
        logging.warning(f"Timeout occurred while requesting {url}. Retrying...")
        raise
    except RequestException as e:
        logging.error(f"Error occurred while requesting {url}: {e}")
        raise

#----------------------------------------------------------------------------------
# Function to scrape data from the lists
def scrape_shelf(session, url, page):
    full_url = f"{url}?page={page}"
    response = make_request(session, full_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    books = []

    #-----------------------------------------
    # Approach 1: Extract using <div> elements with class "elementList"
    for book in soup.find_all('div', class_='elementList'):
        title_elem = book.find('a', class_='bookTitle')
        author_elem = book.find('a', class_='authorName')

        if title_elem and author_elem:
            books.append({
                'title': title_elem.text.strip(),
                'author': author_elem.text.strip(),
                'url': "https://www.goodreads.com" + title_elem['href']
            })

    #-----------------------------------------
    # Approach 2: Extract using <tr> elements with itemtype="http://schema.org/Book"
    for book in soup.find_all('tr', itemtype='http://schema.org/Book'):
        title_elem = book.find('a', class_='bookTitle')
        author_elem = book.find('a', class_='authorName')

        if title_elem and author_elem:
            books.append({
                'title': title_elem.find('span', itemprop='name').text.strip(),
                'author': author_elem.find('span', itemprop='name').text.strip(),
                'url': "https://www.goodreads.com" + title_elem['href']
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
        # Extract mean rate
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
        logging.error(f"Error scraping:\n!!!{url}: {e}")
        return None

#----------------------------------------------------------------------------------
# Main scraping function
def scrape_goodreads_lists(urls, max_pages=30):
    progress = load_progress()
    session = get_session()
    all_books = []
    
    for url in urls:
        books = []
        last_page = progress['urls'].get(url, {}).get('last_page', 0)
        
        for page in range(last_page + 1, max_pages + 1):
            logging.info(f"Scraping:\n{url} - page {page}")
            try:
                page_books = scrape_shelf(session, url, page)
                
                if not page_books:
                    logging.info(f"No more books found on:\n{url} - page {page}\nMoving to next URL.----------------")
                    break
                
                for book in page_books:
                    book_data = scrape_book_page(session, book['url'])
                    if book_data:
                        book_data.update(book)
                        books.append(book_data)
                    else:
                        logging.warning(f"Failed to scrape book:\n!!!{book['url']}")
                    
                # Implement a random delay between 5 to 15 seconds after every page (not at every book)
                time.sleep(random.uniform(5, 15))
                
                # Save progress after each page
                progress['urls'][url] = {'last_page': page, 'books': books}
                save_progress(progress)
                
            except Exception as e:
                logging.error(f"Error scraping:\n!!!{url} - page {page}: {e}")
                time.sleep(60) # Wait a minute before retrying
        
        all_books.extend(books)
    #print(books)
    return all_books

#----------------------------------------------------------------------------------
# Main execution function
def main():

    # Webpages to start the scraping
    urls = ["https://www.goodreads.com/list/show/43374.Classic_Science_Fiction_1930_1939",
            "https://www.goodreads.com/list/show/40744.Classic_Science_Fiction_1940_1949",
            "https://www.goodreads.com/list/show/5152.Classic_Science_Fiction_1950_1959",
            "https://www.goodreads.com/list/show/5158.Classic_Science_Fiction_1960_1969",
            "https://www.goodreads.com/list/show/42069.Classic_Science_Fiction_1970_1979",
            "https://www.goodreads.com/list/show/42417.Classic_Science_Fiction_1980_1989",
            "https://www.goodreads.com/list/show/42875.Classic_Science_Fiction_1990_1999",
            "https://www.goodreads.com/list/show/43319.Classic_Science_Fiction_2000_2009",
            "https://www.goodreads.com/list/show/75182.Science_Fiction_2010_2019",
            "https://www.goodreads.com/list/show/146613.Science_Fiction_2020_2029",
            "https://www.goodreads.com/list/show/79670.Best_Science_Fiction_on_Goodreads_with_fewer_than_100_ratings",
            "https://www.goodreads.com/list/show/78128.Best_Science_Fiction_on_Goodreads_with_between_100_and_999_ratings",
            "https://www.goodreads.com/list/show/77875.Best_Science_Fiction_on_Goodreads_with_between_1000_and_9999_ratings",
            "https://www.goodreads.com/list/show/46769.Popular_Science_Fiction_on_GoodReads_with_between_10000_and_24999_ratings",
            "https://www.goodreads.com/list/show/39287.Popular_Science_Fiction_on_GoodReads_with_between_25000_and_50000_ratings",
            "https://www.goodreads.com/list/show/138257.Popular_Science_Fiction_on_Goodreads_with_between_50000_and_99999_ratings",
            "https://www.goodreads.com/list/show/35776.Most_Popular_Science_Fiction_on_Goodreads",
            "https://www.goodreads.com/list/show/115331.Nineteenth_Century_Science_Fiction",
            "https://www.goodreads.com/list/show/18864.Genetics_in_Science_Fiction",
            "https://www.goodreads.com/list/show/549.Most_Under_rated_Science_Fiction",
            "https://www.goodreads.com/list/show/6228.SF_Masterworks",
            "https://www.goodreads.com/list/show/6934.Science_Fiction_Books_by_Female_Authors",
            "https://www.goodreads.com/list/show/9951.best_hard_science_fiction",
            "https://www.goodreads.com/list/show/17148.Space_Horror",
            "https://www.goodreads.com/list/show/6032.Best_Aliens",
            "https://www.goodreads.com/list/show/485.Best_Books_on_Artificial_Intelligence_",
            "https://www.goodreads.com/list/show/487.Best_of_Cyberpunk",
            "https://www.goodreads.com/list/show/17324.Transhuman_Science_Fiction_",
            "https://www.goodreads.com/list/show/114349.Best_Forgotten_Science_Fiction_of_the_20th_Century",
            "https://www.goodreads.com/list/show/47.Best_Dystopian_and_Post_Apocalyptic_Fiction",
            "https://www.goodreads.com/list/show/7239.Best_Utopian_Dystopian_Fiction",
            "https://www.goodreads.com/list/show/25823",
            "https://www.goodreads.com/list/show/154763.The_Amazing_Colossal_Science_Fiction_Ketchup_Pre_1900s"]

    books = scrape_goodreads_lists(urls, max_pages=30)

    # Create DataFrame with specified column order
    df = pd.DataFrame(books)

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
    
    df.to_csv('./Data/sci-fi_books_PARTIAL_LISTS.csv', index=False, sep=';', encoding='utf-8-sig')
    
    logging.info(f"Scraped {len(books)} books. Data saved to ./Data/sci-fi_books_PARTIAL_LISTS.csv")

    #----------------------------------------------------------------------------------
    # Reading the complete json file and saving it as a CSV file

    # Step 1: Load the JSON file into a Python object
    with open('./Data/scraping_progress.json', 'r') as f:
        data = json.load(f) # Load JSON into a dictionary

    # Step 2: Extract the "books" data from within the "urls" layer
    books_data = []

    # Navigate through the URLs dictionary and extract the "books" lists
    for url_key, url_content in data['urls'].items():
        # Check if "books" is present and is a list
        if 'books' in url_content and isinstance(url_content['books'], list):
            books_data.extend(url_content['books']) # Add the list of books to books_data

    # Step 3: Flatten the books data into a DataFrame
    # Each element in books_data is a book's details
    df_books = pd.json_normalize(books_data)
    
    # Inspect the DataFrame structure
    print("\n",df_books.head()) # View the first few rows to understand the layout
    
    #--------------------------------------------
    # Chose the right columns and their order
    df_books = df_books.reindex(columns=column_order)

    #--------------------------------------------
    # Save the flattened and final DataFrame to a CSV file
    df_books.to_csv('./Data/sci-fi_books_LISTS.csv', index=False, sep=';', encoding='utf-8-sig')

    logging.info(f"Data saved to ./Data/sci-fi_books_LISTS.csv")

#----------------------------------------------------------------------------------
# Execution
if __name__ == "__main__":
    main()