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
    wait=wait_exponential(multiplier=1, min=4, max=600),  # Exponential backoff: starts at 4 seconds, max 600 seconds
    stop=stop_after_attempt(10)  # Stop after 10 attempts
)

# Function to analyse every book
def analyze_book(title, author, year, genres, synopsis, review):
    # Create the prompt with the gathered data
    prompt = f"""
    Carefully consider the plot of the book "{title}" by {author}, published in {year}.
    Please, first provide a single paragraph summarizing and highlighting the most famous and iconic elements of the book, such as:

    Key characters who are often referenced or discussed in popular culture;
    Themes that are central to the story or have a significant impact on the plot;
    Locations and time settings that are unique or memorable to the book;
    Notable creatures (aliens, robots/AI, or others) that are central to the story or have a significant impact on the plot;
    Technologies that are central to the story or have a significant impact on the plot;

    Then, answer the following questions followed by a 1 sentence justification explaining why this choice aligns with the summary, provided information, or well-known aspects of the book.
    For each question, answer exactly in this format: number. alternative: justification

    1. Is the book considered more soft or hard sci-fi?
        (soft: scientific accuracy is not central to the plot or the story emphasizes soft sciences like psychology and sociology, or speculative elements; hard: scientific accuracy is central to the plot or the story emphasizes hard sciences like physics, biology, technology, or realistic scenarios; mixed: elements of both)
    2. When does most of the story take place in relation to the year the book was published?
        (distant past: millennia or more before; far past: centuries before; near past: within a few decades; present; near future: within a few decades; far future: centuries ahead; distant future: millennia or more ahead; multiple timelines; uncertain)
    3. What is the tone of the book?
        (pessimistic: bleak outlook; optimistic: hopeful; neither)
    4. What is the social and political setting of the book?
        (utopic: ideal society; dystopic: oppressive society; neither)
    5. Is most of the story set on Earth?
        (yes or no)
    6. Is the story set in a post-apocalyptic world (after a big civilization-collapsing event)?
        (yes, no, or somewhat)
    7. Are there any depictions or mentions of non-terrestrial life forms (e.g., aliens, extraterrestrial organisms, creatures not originating from Earth, even if non-sentient) or alien technology in the story?
        (yes or no)
    8. How are the non-terrestrial life forms generally depicted in the story?
        (good: friendly, virtuous, helpful, or heroic; bad: hostile, villainous, antagonistic, or threatening; nuanced: complex or showing both positive and negative traits; irrelevant: minor role, or not significantly affecting the plot; not applicable: no non-terrestrial life forms present)
    9. Are there any depictions of robots or artificial intelligences in the story?
        (yes or no)
    10. How are the robots or artificial intelligences generally depicted?
        (good: friendly, benign, virtuous, helpful, or heroic; bad: hostile, malignant, villainous, antagonistic, or threatening; nuanced: complex or showing both positive and negative traits; irrelevant: minor role, or not significantly affecting the plot; not applicable: no robots or artificial intelligences present)
    11. How is technology and science depicted in the story?
        (good: beneficial, advancing society, solving problems, or optimistic; bad: harmful, causing problems, being misused, or pessimistic; mixed: complex, with both benefits and drawbacks; neutral: present but not central to the story's themes or impact)
    12. What is the gender of the protagonist?
        (male, female, or other)
    13. Does the story explicitly address, critique, or reflect specific social issues relevant to the time of publication (e.g., inequality, war, discrimination, political oppression)?
        (yes, no, or somewhat)
    14. Is there an environmental or ecological message in the book?
        (yes, no, or somewhat)

    To help answer the questions, consider the genres the book fits in: {genres}.
    This short synopsis: {synopsis}
    And this partial review: {review}
    """
    
    # Call the OpenAI API with the crafted prompt
    ChatCompletion = client.chat.completions.create(
        messages = [
            {"role": "system", "content": "You are a helpful assistant and scholar of comparative sci-fi literature who analyzes book plots based on your own knowledge and provided information."},
            {"role": "user", "content": prompt}
        ],
        #model = "gpt-4o-mini-2024-07-18",
        model = "gpt-4o",
        max_tokens = 600,  # Adjust as necessary based on the detail needed
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

    # Summarizing paragraph
    paragraph = []

    # Answer to the questions
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
    tech_sci = []
    protagonist = []
    social = []
    enviromental = []
    
    # Justification to the answer
    soft_hard_just = []
    time_just = []
    tone_just = []
    setting_just = []
    on_earth_just = []
    post_apocalyptic_just = []
    aliens_just = []
    aliens_are_just = []
    robots_ai_just = []
    robots_ai_are_just = []
    tech_sci_just = []
    protagonist_just = []
    social_just = []
    enviromental_just = []

    # Load existing progress if the file exists
    output_file = 'AI_answers_to_sci-fi_books.csv'
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
        decade = book['decade']
        pages = book['pages']
        series = book['series']
        genres = book['genres']
        synopsis = book['synopsis']
        review = book['review']
        url = book['url']

        try:
            # Get the AI's answers for the book
            AI_answers = analyze_book(title, author, year, genres, synopsis, review)

            # Split answers into a list
            answers = []
            justifications = []
            lines = AI_answers.split('\n')            

            # Process the first line differently
            paragraph_text = lines[0]

            # Process each line
            for line in lines[2:]: # Start on the third line (first is paragraph and second is empty)
                # Split at the first occurrence of ". " to separate the number
                parts_number_text = line.split('. ', 1)
                
                # Further split at the first occurrence of ": " to separate answer and justification
                parts_answer_just = parts_number_text[1].split(': ', 1)
                
                # Check if the split was successful, i.e., there are exactly two parts
                if len(parts_answer_just) == 2:
                    answers.append(parts_answer_just[0].strip()) # Append the word after the number and period
                    justifications.append(parts_answer_just[1].strip()) # Append the text after the colon

            # Append answers to respective lists
            if (len(answers) == 14) & (len(justifications) == 14):
                paragraph.append(paragraph_text)

                soft_hard.append(answers[0])
                soft_hard_just.append(justifications[0])

                time.append(answers[1])
                time_just.append(justifications[1])

                tone.append(answers[2])
                tone_just.append(justifications[2])

                setting.append(answers[3])
                setting_just.append(justifications[3])

                on_earth.append(answers[4])
                on_earth_just.append(justifications[4])

                post_apocalyptic.append(answers[5])
                post_apocalyptic_just.append(justifications[5])

                aliens.append(answers[6])
                aliens_just.append(justifications[6])

                aliens_are.append(answers[7])
                aliens_are_just.append(justifications[7])

                robots_ai.append(answers[8])
                robots_ai_just.append(justifications[8])

                robots_ai_are.append(answers[9])
                robots_ai_are_just.append(justifications[9])

                tech_sci.append(answers[10])
                tech_sci_just.append(justifications[10])

                protagonist.append(answers[11])
                protagonist_just.append(justifications[11])

                social.append(answers[12])
                social_just.append(justifications[12])

                enviromental.append(answers[13])
                enviromental_just.append(justifications[13])
            else:
                logging.warning(f"Unexpected number of answers for book: {title}")
                paragraph.extend([None] * 29)

            # Save progress after each book
            progress_df = pd.DataFrame({
                'title': [title],
                'author': [author],
                'year': [year],

                'paragraph': [paragraph[-1]],

                '1 soft hard': [soft_hard[-1]],
                'justifying soft hard': [soft_hard_just[-1]],

                '2 time': [time[-1]],
                'justifying time': [time_just[-1]],

                '3 tone': [tone[-1]],
                'justifying tone': [tone_just[-1]],

                '4 setting': [setting[-1]],
                'justifying setting': [setting_just[-1]],

                '5 on Earth': [on_earth[-1]],
                'justifying on Earth': [on_earth_just[-1]],

                '6 post apocalyptic': [post_apocalyptic[-1]],
                'justifying post apocalyptic': [post_apocalyptic_just[-1]],

                '7 aliens': [aliens[-1]],
                'justifying aliens': [aliens_just[-1]],

                '8 aliens are': [aliens_are[-1]],
                'justifying aliens are': [aliens_are_just[-1]],

                '9 robots and AI': [robots_ai[-1]],
                'justifying robots and AI': [robots_ai_just[-1]],

                '10 robots and AI are': [robots_ai_are[-1]],
                'justifying robots and AI are': [robots_ai_are_just[-1]],

                '11 tech and science': [tech_sci[-1]],
                'justifying tech and science': [tech_sci_just[-1]],

                '12 protagonist': [protagonist[-1]],
                'justifying protagonist': [protagonist_just[-1]],

                '13 social issues': [social[-1]],
                'justifying social issues': [social_just[-1]],

                '14 enviromental': [enviromental[-1]],
                'justifying enviromental': [enviromental_just[-1]],

                'decade': [decade],
                'pages': [pages],
                'series': [series],
                'genres': [genres],
                'synopsis': [synopsis],
                'review': [review],
                'url': [url],
            })
            processed_df = pd.concat([processed_df, progress_df], ignore_index=True)
            processed_df.to_csv(output_file, index=False, sep=';')
            logging.info(f"Progress saved for book: {title}")

        except Exception as e:
            logging.error(f"Failed to analyze book: {title}. Error: {e}")

    return processed_df

#----------------------------------------------------------------------------------
# Main execution
df = pd.read_csv("top_books_TEST.csv", sep=';')
processed_df = ask_to_AI(df)

print(processed_df.info())
#print(processed_df.head())

processed_df.to_csv('AI_ANSWERS_TO_sci-fi_books.csv', index=False, sep=';')
print(f"Data saved to AI_ANSWERS_TO_sci-fi_books.csv")