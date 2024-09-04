import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from requests.exceptions import RequestException
import logging

#----------------------------------------------------------------------------------
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the OpenAI API key
load_dotenv(dotenv_path='../KEYs/My_OPENAI_API_Key.env')
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

#----------------------------------------------------------------------------------
# Function to call OpenAI API with retry logic
@retry(
    retry=retry_if_exception_type((RequestException, Exception)),  # Retry on API errors or network issues
    wait=wait_exponential(multiplier=1, min=4, max=60),  # Exponential backoff: starts at 4 seconds, max 60 seconds
    stop=stop_after_attempt(5)  # Stop after 5 attempts
)

# Function to analyse every book
def analyze_book(title, author, year, synopsis):
    # Create the prompt with the gathered data
    prompt = f"""
    Based on the following information about the book "{title}" by {author}, published in {year}, answer the following questions using the options provided:

    1. Is the book considered more soft or hard sci-fi?
        (soft: focuses on character, society, action, or speculation; hard: focuses on scientific accuracy; mixed: elements of both)
    2. When does most of the story take place in relation to the year the book was published?
        (far past: centuries or more before; near past: within a few decades; present; near future: within a few decades; far future: centuries or more ahead; multiple timelines; uncertain)
    3. What is the tone of the book?
        (pessimistic: bleak outlook; optimistic: hopeful; neither)
    4. What is the social and political setting of the book?
        (utopic: ideal society; dystopic: oppressive society; neither)
    5. Is most of the story set on Earth?
        (yes or no)
    6. Is the story set in a post-apocalyptic world?
        (yes, no, somewhat)
    7. Are there any alien life (life which does not originate from Earth) in the story?
        (yes or no)
    8. How are the aliens depicted?
        (good: friendly; bad: hostile; both/mixed: nuanced; not applicable)
    9. Are there any robots or artificial intelligences in the story?
        (yes or no)
    10. How are the robots or artificial intelligences depicted?
        (good: friendly; bad: hostile; both/mixed: nuanced; not applicable)
    11. What is the gender of the protagonist?
        (male, female, other)
    12. Can the story be seen as a commentary on social issues of the time of publication?
        (yes, no, somewhat)
    13. Is there an environmental message in the book?
        (yes, no, somewhat)

    Please, answer with only the question number and one of the options for each question. (do not repeat the questions)

    If you know the book well, use your own knowledge first, but also consider this short synopsis: {synopsis}
"""
    
    # Call the OpenAI API with the crafted prompt
    ChatCompletion = client.chat.completions.create(
        messages = [
            {"role": "system", "content": "You are a helpful assistant and schoolar that analyzes book plots based on provided information and your own knowledge about the books."},
            {"role": "user", "content": prompt}
        ],
        #model = "gpt-4o-mini-2024-07-18",
        model = "gpt-4o",
        max_tokens = 500,  # Adjust as necessary based on the detail needed
        temperature = 0.3  # Adjust for creativity vs. factual response balance
    )
    
    # Extract and print the response
    answer = ChatCompletion.choices[0].message.content
    print(f'\n"{title}" by {author}, {year}')
    print(answer)

    return answer

#----------------------------------------------------------------------------------
# Function to process each book and save progress incrementally
def ask_to_AI(df):
    # Lists to store the answers for each book
    soft_hard = []
    time = []
    tone = []
    setting = []
    on_earth = []
    post_apocalyptic = []
    aliens = []
    aliens_are = []
    robots_ai = []
    robots_ai_are = []
    gender = []
    social = []
    enviromental = []

    # Load existing progress if the file exists
    output_file = 'AI-answers_to_sci-fi_books.csv'
    if os.path.exists(output_file):
        processed_df = pd.read_csv(output_file, sep=';')
        processed_titles = set(processed_df['title'])
    else:
        processed_df = pd.DataFrame()
        processed_titles = set()

    for _, book in df.iterrows():
        # Skip already processed books
        if book['title'] in processed_titles:
            continue

        # Extract book details
        title = book['title']
        author = book['author']
        year = book['year']
        synopsis = book['synopsis']
        #url = book['url']

        try:
            # Get the AI's answers for the book
            AI_answers = analyze_book(title, author, year, synopsis)
            
            # Split answers into a list
            answers = []
            lines = AI_answers.split('\n')
            for line in lines:
                parts = line.split('. ', 1)
                if len(parts) == 2:
                    answers.append(parts[1].strip())

            # Append answers to respective lists
            if len(answers) == 13:
                soft_hard.append(answers[0])
                time.append(answers[1])
                tone.append(answers[2])
                setting.append(answers[3])
                on_earth.append(answers[4])
                post_apocalyptic.append(answers[5])
                aliens.append(answers[6])
                aliens_are.append(answers[7])
                robots_ai.append(answers[8])
                robots_ai_are.append(answers[9])
                gender.append(answers[10])
                social.append(answers[11])
                enviromental.append(answers[12])
            else:
                logging.warning(f"Unexpected number of answers for book: {title}")
                soft_hard.extend([None] * 13)

            # Save progress after each book
            progress_df = pd.DataFrame({
                'title': [title],
                'author': [author],
                'year': [year],
                'synopsis': [synopsis],
                #'url': [url],

                'soft hard': [soft_hard[-1]],
                'time': [time[-1]],
                'tone': [tone[-1]],
                'setting': [setting[-1]],
                'on Earth': [on_earth[-1]],
                'post apocalyptic': [post_apocalyptic[-1]],
                'aliens': [aliens[-1]],
                'aliens are': [aliens_are[-1]],
                'robots and AI': [robots_ai[-1]],
                'robots and AI are': [robots_ai_are[-1]],
                'gender': [gender[-1]],
                'social issues': [social[-1]],
                'enviromental': [enviromental[-1]]
            })
            processed_df = pd.concat([processed_df, progress_df], ignore_index=True)
            processed_df.to_csv(output_file, index=False, sep=';')
            logging.info(f"Progress saved for book: {title}")

        except Exception as e:
            logging.error(f"Failed to analyze book: {title}. Error: {e}")

    return processed_df

#----------------------------------------------------------------------------------
# Main execution
df = pd.read_csv("sci-fi_top_books.csv", sep=';')
processed_df = ask_to_AI(df)

print(processed_df.info())
print(processed_df.head())

processed_df.to_csv('AI-answers_to_sci-fi_books_final.csv', index=False, sep=';')
print(f"Data saved to AI-answers_to_sci-fi_books_final.csv")