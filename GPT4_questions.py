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
    retry=retry_if_exception_type((RequestException, Exception)), # Retry on API errors or network issues
    wait=wait_exponential(multiplier=1, min=4, max=600), # Exponential backoff: starts at 4 seconds, max 600 seconds
    stop=stop_after_attempt(10) # Stop after 10 attempts
)

# Function to analyse every book
def analyze_book(title, author, year, synopsis, review, genres):
    # Create the prompt with the gathered data
    prompt = f"""
    Carefully consider the plot of the book "{title}" by {author}, published in {year}, focusing on key elements that will help answer the following questions. 
    
    **Output Formatting Instructions**:
    Follow this exact format!

    Provide a concise paragraph summarizing the most iconic and well-known elements of the book, including:
    - Themes central to the story or that significantly impact the plot;
    - Memorable locations or time settings unique to the story;
    - The main character, if there is one, and secondary characters;
    - Important creatures (e.g., aliens, robots, AIs) central to the story;
    - Notable technologies that play a key role in the plot.

    After the summary, leave one blank line and provide one answer per line in the following format:
    Question number, followed by a period, then the selected alternative, followed by a colon, and then the detailed but short justification in a single sentence.
    Example:
    1. Hard: The story emphasizes hard sciences like physics and biology.
    2. Near future: The setting is within a few decades ahead of the publication date.
    3. Pessimistic: The story maintains a bleak outlook with occasional hopeful moments.

    Important Notes:
    Ensure there are no line breaks, extra spaces, or symbols between answers.
    If no clear summary is possible, provide a brief explanation before answering.

    **Questions**:
    1. Is the book considered more soft or hard sci-fi?
        Very soft: scientific accuracy is minimal or irrelevant, leaning towards fantasy or speculation;
        Soft: emphasis on soft or social sciences such as psychology, sociology, or philosophy, with less focus on scientific rigor;
        Mixed: balances elements of both soft and hard sci-fi equally;
        Hard: emphasis on hard or natural sciences like physics, astronomy, biology, or technology, with more focus on scientific plausibility;
        Very hard: scientific accuracy is essential, leaning more on realism or extrapolation from known science.
    2. When does most of the story take place in relation to the year the book was published?
        Distant past: millennia or more before; 
        Far past: centuries before; 
        Near past: within a few decades before; 
        Present: within a few years; 
        Near future: within a few decades ahead; 
        Far future: centuries ahead; 
        Distant future: millennia or more ahead; 
        Multiple timelines; 
        Uncertain.
    3. What is the tone of the story?
        Very optimistic: overwhelmingly positive, uplifting, hopeful; 
        Optimistic: positive outlook but with moments of pessimism; 
        Balanced: mix of positive and negative tones without leaning towards one; 
        Pessimistic: negative outlook but with moments of optimism; 
        Very pessimistic: overwhelmingly negative, bleak, hopeless.
    4. How is the social and political setting depicted in the story?
        Utopic: ideal or perfect society;
        Leaning utopic: mostly positive and desirable but with some flaws;
        Balanced: mix of positive and negative elements, or an ordinary societal view;
        Leaning dystopic: mostly negative and undesirable but with redeeming aspects;
        Dystopic: bleak, authoritarian, and oppressive;
        Uncertain: social and political setting is either unclear or not a major focus of the story.
    5. Is most of the story set on Earth?
        Yes;
        No.
    6. Is the story set in a post-apocalyptic world (following a major civilization-collapse event)?
        Yes: clear post-apocalyptic state;
        Somewhat: just some elements are present, the collapse is partial or local;
        No: not set in a post-apocalyptic world.
    7. Are there any depictions or mentions of non-terrestrial life forms (e.g., aliens, extraterrestrial organisms, creatures not originating from Earth, even if non-sentient) or alien technology in the story?
        Yes;
        No.
    8. How are the non-terrestrial life forms generally depicted in the story?
        Good: friendly, virtuous, helpful, or heroic; 
        Leaning good: generally positive or benign but with flaws or minor conflicts; 
        Ambivalent: showing both positive and negative traits, morally ambiguous or multifaceted; 
        Leaning bad: generally antagonistic or threatening but not entirely villainous; 
        Bad: hostile, villainous, antagonistic, or threatening; 
        Uncertain: lack of (moral) characterization or minimal plot relevance; 
        Not applicable: no non-terrestrial life forms present.
    9. Are there any depictions of robots or artificial intelligences in the story? (just automatic systems or programs do not count)
        Yes;
        No.
    10. How are the robots or artificial intelligences generally depicted?
        Good: friendly, virtuous, helpful, or heroic; 
        Leaning good: generally positive or benign but with flaws or minor conflicts; 
        Ambivalent: showing both positive and negative traits, morally ambiguous or multifaceted; 
        Leaning bad: generally antagonistic or threatening but not entirely villainous; 
        Bad: hostile, villainous, antagonistic, or threatening; 
        Uncertain: lack of (moral) characterization or minimal plot relevance; 
        Not applicable: no robots or artificial intelligences present.
    11. Is there a single protagonist or main character?
        Yes; 
        No.
    12. What is the gender of the single protagonist or main character?
        Male; 
        Female; 
        Other: the gender is ambiguous, fluid, or neither male nor female;
        Non-human: the central character is non-human; 
        Not applicable: no clear single main character.
    13. How are technology and science depicted in the story?
        Good: optimistic and beneficial portrayal; 
        Leaning good: mostly positive but with minor issues; 
        Ambivalent: balanced view with both positive and negative consequences; 
        Leaning bad: largely negative portrayal, but with redeeming features; 
        Bad: pessimistic, harmful or destructive portrayal; 
        Uncertain: lack of characterization or minimal plot relevance.
    14. Is virtual reality or immersive digital environments (e.g., simulations, augmented reality) present in the story?
        Yes: central or significant role in the plot;
        Somewhat: some form of it, minor or background role;
        No: not present.
    15. How central is the critique or reflection of specific social issues (e.g., inequality, war, discrimination, political oppression) to the story?
        Core: main driver of the plot or key theme;
        Major: significant role but secondary theme;
        Minor: subtle role or minimal theme;
        Absent: not present.
    16. How central is an ecological or environmental message to the story?
        Core: main driver of the plot or key theme;
        Major: significant role but secondary theme;
        Minor: subtle role or minimal theme;
        Absent: not present.

    When answering, consider how the following synopsis, review, and genres may provide relevant context for each question.

    Book synopsis: {synopsis}
    Partial review: {review}
    Genres the book fits in: {genres}
    """
    
    # Call the OpenAI API with the crafted prompt
    ChatCompletion = client.chat.completions.create(
        messages = [
            {"role": "system", "content": "You are a helpful assistant and scholar of comparative sci-fi literature who analyzes book plots based on your own knowledge and provided information."},
            {"role": "user", "content": prompt}
        ],
        #model = "gpt-4o-mini-2024-07-18",
        model = "gpt-4o-2024-08-06",
        #model = "gpt-4o",
        max_tokens = 700, # Adjust based on the detail needed
        temperature = 0.2 # Adjust for factual response vs. creativity balance
    )
    
    # Extract and print the response
    answer = ChatCompletion.choices[0].message.content
    print(f'\n"{title}" by {author}, {year}')
    #print(prompt)
    print(answer)

    return answer

#----------------------------------------------------------------------------------
# Function to process each book and save progress incrementally
def ask_to_AI(df):
    # Lists to store the complete answer the AI give and its parts for each book

    # Complete AI answer
    complete_answer = []

    # Summarizing paragraph
    paragraph = []

    # Answers to the questions
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
    protagonist = []
    protagonist_is = []
    tech_sci = []
    virtual = []
    social = []
    enviromental = []
    
    # Justifications to the answers given
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
    protagonist_just = []
    protagonist_is_just = []
    tech_sci_just = []
    virtual_just = []
    social_just = []
    enviromental_just = []

    #----------------------------------------
    # Load existing progress if the file exists
    output_file = './Data/AI_answers_to_sci-fi_books_test.csv'
    if os.path.exists(output_file):
        df_processed = pd.read_csv(output_file, sep=';', encoding="utf-8-sig")

        processed_books = set(df_processed['url'])
    else:
        df_processed = pd.DataFrame()
        processed_books = set()
    for _, book in df.iterrows():
        # Skip already processed books
        if book['url'] in processed_books:
            continue

        # Extract book details
        title = book['title']
        author = book['author']
        year = int(book['year'])
        decade = int(book['decade'])
        #pages = book['pages']
        rate = float(book['rate'])
        ratings = int(book['ratings'])
        series = book['series']
        genres = book['genres']
        synopsis = book['synopsis']
        review = book['review']
        url = book['url']

        #----------------------------------------
        try:
            # Get the AI's answers for the book
            AI_answers = analyze_book(title, author, year, synopsis, review, genres)
            #print("\n",AI_answers)

            # Split answers into a list
            answers = []
            justifications = []
            lines = AI_answers.split('\n')            

            # Process the first line differently
            paragraph_text = lines[0]

            # Sometimes the AI output may or may not have a break line or even a paragraph, or the text may be all in one line
            # You can go along with just the paragraph and the answers with no break line between them
            # Problematic books/rows will be nulled an excluded later in the program
            # But you will need to rerun the program to try again to get (only) those books/rows
            #----------------------------------------
            # If there is a paragraph and NO break line separating it from the answers
            if len(lines) == 17:
                # Process each line
                for line in lines[1:]: # Start from the second line (first is paragraph)
                    # Split at the first occurrence of ". " to separate the number
                    parts_number_text = line.split('. ', 1)
                    
                    # Further split at the first occurrence of ": " to separate answer and justification
                    parts_answer_just = parts_number_text[1].split(': ', 1)
                    
                    # Check if the split was successful, i.e., there are exactly two parts
                    if len(parts_answer_just) == 2:
                        answers.append(parts_answer_just[0].strip()) # Append the word after the number and period
                        justifications.append(parts_answer_just[1].strip()) # Append the text after the colon

            #----------------------------------------
            # If there is a paragraph and a break line separating it from the answers
            elif len(lines) == 18:
                # Process each line
                for line in lines[2:]: # Start from the third line (first is paragraph and second is empty)
                    # Split at the first occurrence of ". " to separate the number
                    parts_number_text = line.split('. ', 1)
                    
                    # Further split at the first occurrence of ": " to separate answer and justification
                    parts_answer_just = parts_number_text[1].split(': ', 1)
                    
                    # Check if the split was successful, i.e., there are exactly two parts
                    if len(parts_answer_just) == 2:
                        answers.append(parts_answer_just[0].strip()) # Append the word after the number and period
                        justifications.append(parts_answer_just[1].strip()) # Append the text after the colon

            #----------------------------------------
            # Append answers to respective lists
            complete_answer.append(AI_answers)

            if (len(answers) == 16) & (len(justifications) == 16):
                
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

                protagonist.append(answers[10])
                protagonist_just.append(justifications[10])

                protagonist_is.append(answers[11])
                protagonist_is_just.append(justifications[11])

                tech_sci.append(answers[12])
                tech_sci_just.append(justifications[12])

                virtual.append(answers[13])
                virtual_just.append(justifications[13])

                social.append(answers[14])
                social_just.append(justifications[14])

                enviromental.append(answers[15])
                enviromental_just.append(justifications[15])
            else:
                logging.warning(f"Unexpected number of answers for book: {title}\nFound {len(answers)} answers and {len(justifications)} justifications.")

                # Something went wrong and better fill a row of None in each list
                paragraph.append(paragraph_text)

                soft_hard.append(None)
                soft_hard_just.append(None)

                time.append(None)
                time_just.append(None)

                tone.append(None)
                tone_just.append(None)

                setting.append(None)
                setting_just.append(None)

                on_earth.append(None)
                on_earth_just.append(None)

                post_apocalyptic.append(None)
                post_apocalyptic_just.append(None)

                aliens.append(None)
                aliens_just.append(None)

                aliens_are.append(None)
                aliens_are_just.append(None)

                robots_ai.append(None)
                robots_ai_just.append(None)

                robots_ai_are.append(None)
                robots_ai_are_just.append(None)

                protagonist.append(None)
                protagonist_just.append(None)

                protagonist_is.append(None)
                protagonist_is_just.append(None)

                tech_sci.append(None)
                tech_sci_just.append(None)

                virtual.append(None)
                virtual_just.append(None)

                social.append(None)
                social_just.append(None)

                enviromental.append(None)
                enviromental_just.append(None)

            #----------------------------------------
            # One-row dataframe to save the progress in the present book/row
            df_progress = pd.DataFrame({
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

                '11 protagonist': [protagonist[-1]],
                'justifying protagonist': [protagonist_just[-1]],

                '12 protagonist is': [protagonist_is[-1]],
                'justifying protagonist is': [protagonist_is_just[-1]],

                '13 tech and science': [tech_sci[-1]],
                'justifying tech and science': [tech_sci_just[-1]],

                '14 virtual': [virtual[-1]],
                'justifying virtual': [virtual_just[-1]],

                '15 social issues': [social[-1]],
                'justifying social issues': [social_just[-1]],

                '16 enviromental': [enviromental[-1]],
                'justifying enviromental': [enviromental_just[-1]],

                'complete answer': [complete_answer[-1]],

                'decade': [decade],
                #'pages': [pages],
                'rate': [rate],
                'ratings': [ratings],
                'series': [series],
                'genres': [genres],
                'synopsis': [synopsis],
                'review': [review],
                'url': [url]
            })

            # Concatenate the one-row dataframe with the big dataframe with all anterior books/rows
            df_processed = pd.concat([df_processed, df_progress], ignore_index=True)
            df_processed.to_csv(output_file, index=False, sep=';')
            logging.info(f"Progress saved for book: {title}")

        except Exception as e:
            logging.error(f"Failed to analyze book: {title}. Error: {e}\nPlease, try again later.")

    return df_processed

#----------------------------------------------------------------------------------
# Main execution

#------------------------------------------
# Load book data to be sent to the AI
df = pd.read_csv("./Data/top_books_TEST.csv", sep=';', encoding="utf-8-sig")
#df = pd.read_csv("./Data/top_sci-fi_books_200_PER_DECADE.csv", sep=';', encoding="utf-8-sig")

# Ask the AI
df_processed = ask_to_AI(df)

# Retyping columns of the processed dataframe
df_processed['year'] = df_processed['year'].astype(int)
df_processed['decade'] = df_processed['decade'].astype(int)
df_processed['rate'] = df_processed['rate'].astype(float)
df_processed['ratings'] = df_processed['ratings'].astype(int)

#------------------------------------------
# Sometimes the AI output is not formatted right
# This will exclude wrong rows (null on paragraph or some other columns just to be sure)
# You will need to rerun the program at least once to get all the books/rows but it will keep the progress
df_processed = df_processed.dropna(axis=0, subset=['paragraph', 'justifying on Earth', '11 protagonist', 'justifying enviromental'], how = 'any', ignore_index=True)

# This will exclude books/rows without paragraphs but you will need to rerun the program at least once to get all the books
"""for index, row in df_processed.iterrows():
    if row['paragraph'][0] == "1":
        #df_processed = df_processed.drop(labels=row['paragraph'], axis=0)
        df_processed = df_processed.drop(index)"""

df_processed = df_processed.sort_values(by = ['decade', 'year', 'author', 'title'], ascending=True)

#------------------------------------------
print('\n',df_processed.info())
#print(df_processed.head())

size_in = df.shape[0] # Number of rows
size_out = df_processed.shape[0] # Number of rows
missing_books = size_in - size_out # Difference in number of rows

print(f"If the number of books missing ({missing_books}) is higher than 0, rerun this program until it is 0 AND there are no more WARNINGS.")

#------------------------------------------
#df_processed.to_csv('./Data/AI_ANSWERS_TO_sci-fi_books_test.csv', index=False, sep=';', encoding='utf-8-sig')
df_processed.to_csv('./Data/AI_ANSWERS_TO_sci-fi_books.csv', index=False, sep=';', encoding='utf-8-sig')
print(f"Data saved to ./Data/AI_ANSWERS_TO_sci-fi_books.csv")