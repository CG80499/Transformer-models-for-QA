import os
import openai

openai.api_key = "YOUR_API_KEY"

interesting_questions = ["when was texas admitted to the union",
                        "when was texas admitted to the confederacy",
                        "which french king was killed in the french revolution",
                        "what is the tallest mountain in the world",
                        "when was the act of union between england and scotland signed",
                        "In mathematics, who invented calculus",
                        "In physics, who discovered special relativity",
                        'when did the coronation of queen elizabeth takes place',
                        "what is the largest country in the world",
                        "who was the last Tsar",
                        'which soviet leader gets his name from "man of steel"',
                        "what is the fastest land animal",
                        "what is longest river in the world",
]
answers = []
def process(q):
    return q[0].upper()+q[1:]+"?"
for q in interesting_questions:
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=process(q),
        temperature=0,
        max_tokens=32,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    answers.append(response.choices[0].text.replace("\n", ""))
print(answers)