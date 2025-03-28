from text_summary import *

def text_summary(num):
    with open(f'./samples/Sample{num}.txt', 'r', encoding='utf-8') as file:
        # Read the file's content
        content = file.read()

    summary = text_summary_bart(content)
    summary1 = text_summary_T5(content)
    summary2 = text_summary_pegasus(content)
    with open(f'./samples/summary{num}_BART.txt', 'w', encoding='utf-8') as file:
        file.write(summary)
    with open(f'./samples/summary{num}_T5.txt', 'w', encoding='utf-8') as file:
        file.write(summary1)
    with open(f'./samples/summary{num}_Pegasus.txt', 'w', encoding='utf-8') as file:
        file.write(summary2)


text_summary(1)
text_summary(2)