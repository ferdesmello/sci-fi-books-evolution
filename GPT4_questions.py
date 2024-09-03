import os
from openai import OpenAI

#----------------------------------------------------------------------------------
# Seting the API key
key_file = open('My OAI API Key.txt', 'r')
MY_OAI_API_KEY = key_file.readlines()[0]
key_file.close()

client = OpenAI(
    api_key = MY_OAI_API_KEY
)

#----------------------------------------------------------------------------------
# Function to analyse every book
def analyze_book(title, author, year, synopsis):
    # Create the prompt with the gathered data
    prompt = f"""
    Based on the following information about the book "{title}" by {author}, published in {year}, determine the following:
    1. Is the book considered more soft or hard sci-fi? (soft, hard, or mixed)
    2. When does the story take place in relation to the year the book was published? (far past, near past, present, near future, far future, or uncertain/other)
    3. What is the tone of the book? (pessimistic, optimistic, or neither)
    4. What is the social setting of the book? (utopic, dystopic, or neither)
    5. Is most of the story set on Earth? (yes or no)
    6. Is the story set in a post-apocalyptic world? (yes, no, or somewhat)
    7. Are there aliens in the story? (yes or no)
    8. How are the aliens depicted? (good, bad, both/mixed, or not applicable)
    9. Are there robots or artificial intelligences in the story? (yes or no)
    10. How are the robots or artificial intelligences depicted? (good, bad, both/mixed, or not applicable)
    11 What is the gender of the protagonist? (male, female, or other)
    12 Can the story be seen as a commentary on social issues of the time of publication? (yes, no, or somewhat)
    13 Is there an environmental message in the book? (yes, no, or somewhat)
    Please, answer with only the words given as options for the questions.
    Book synopsis: {synopsis}"""
    
    # Call the OpenAI API with the crafted prompt
    ChatCompletion = client.chat.completions.create(
        messages = [
            {"role": "system", "content": "You are a helpful assistant that analyzes books based on provided information and your own knowledge about the book."},
            {"role": "user", "content": prompt}
        ],
        #model = "gpt-4o-mini-2024-07-18",
        model = "gpt-4o",
        max_tokens = 500,  # Adjust as necessary based on the detail needed
        temperature = 0.7  # Adjust for creativity vs. factual response balance
    )
    
    # Extract and print the response
    answer = ChatCompletion.choices[0].message.content
    print(f'"{title}" by {author}, {year}')
    print(answer)

#----------------------------------------------------------------------------------
# Example usage with data you have extracted

title = "Ready Player One"
author = "Ernest Cline"
synopsis = "IN THE YEAR 2044, reality is an ugly place. The only time teenage Wade Watts really feels alive is when he's jacked into the virtual utopia known as the OASIS. Wade's devoted his life to studying the puzzles hidden within this world's digital confines, puzzles that are based on their creator's obsession with the pop culture of decades past and that promise massive power and fortune to whoever can unlock them. But when Wade stumbles upon the first clue, he finds himself beset by players willing to kill to take this ultimate prize. The race is on, and if Wade's going to survive, he'll have to win\u2014and confront the real world he's always been so desperate to escape."
year = "2011"

title = "The War of the Worlds"
author = "H.G. Wells"
synopsis = "When an army of invading Martians lands in England, panic and terror seize the population. As the aliens traverse the country in huge three-legged machines, incinerating all in their path with a heat ray and spreading noxious toxic gases, the people of the Earth must come to terms with the prospect of the end of human civilization and the beginning of Martian rule.Inspiring films, radio dramas, comic-book adaptations, television series and sequels,The War of the Worlds is a prototypical work of science fiction which has influenced every alien story that has come since, and is unsurpassed in its ability to thrill, well over a century since it was first published."
year = "1898"

title = "Dune"
author = "Frank Herbert"
synopsis = "Set on the desert planet Arrakis, Dune is the story of the boy Paul Atreides, heir to a noble family tasked with ruling an inhospitable world where the only thing of value is the    spice    melange, a drug capable of extending life and enhancing consciousness. Coveted across the known universe, melange is a prize worth killing for...When House Atreides is betrayed, the destruction of Paul   s family will set the boy on a journey toward a destiny greater than he could ever have imagined. And as he evolves into the mysterious man known as Muad   Dib, he will bring to fruition humankind   s most ancient and unattainable dream."
year = "1965"

title = "Life, the Universe and Everything"
author = "Douglas Adams"
synopsis = "Now celebrating the 42nd anniversary of\u00a0 The Hitchhiker\u2019s Guide to the Galaxy, \u00a0soon to be a Hulu original series!\u201cWild satire . . . The feckless protagonist, Arthur Dent, is reminiscent of Vonnegut heroes.\u201d\u2014 Chicago TribuneThe unhappy inhabitants of planet Krikkit are sick of looking at the night sky above their heads\u2014so they plan to destroy it. The universe, that is. Now only five individuals stand between the killer robots of Krikkit and their goal of total annihilation.They are Arthur Dent, a mild-mannered space and time traveler who tries to learn how to fly by throwing himself at the ground and missing; Ford Prefect, his best friend, who decides to go insane to see if he likes it; Slartibartfast, the indomitable vice president of the Campaign for Real Time, who travels in a ship powered by irrational behavior; Zaphod Beeblebrox, the two-headed, three-armed ex-president of the galaxy; and Trillian, the sexy space cadet who is torn between a persistent Thunder God and a very depressed Beeblebrox.How will it all end? Will it end? Only this stalwart crew knows as they try to avert \u201cuniversal\u201d Armageddon and save life as we know it\u2014and don\u2019t know it!\u201cAdams is one of those rare an author who, one senses, has as much fun writing as one has reading."
year = "1982"

#----------------------------------------------------------------------------------
answer = analyze_book(title, author, year, synopsis)

print(answer)