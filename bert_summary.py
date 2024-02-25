from summarizer import Summarizer

# Load your text from a file or provide it as a string
with open("1953_L_1.txt", "r") as file:
    lines = file.readlines()

# Process the data to separate sentences and labels
data_by_label = {}
current_label = None
current_text = []
for line in lines:
    line = line.strip()
    if line:
        if '\t' in line:
            # Assume label is separated by tab from the text
            label, text = line.split('\t', 1)
            current_label = label.strip()
            current_text.append(text.strip())
        else:
            current_text.append(line)
    elif current_label:
        data_by_label.setdefault(current_label, []).append(' '.join(current_text))
        current_text = []

# Initialize the Summarizer
summarizer = Summarizer()

# Specify the number of sentences in the summary
num_sentences = 3  # You can adjust this value as needed

# Generate and print summaries based on labels
for label, text_list in data_by_label.items():
    text = ' '.join(text_list)
    summary = summarizer(text, num_sentences=num_sentences)
    print(f"Summary for label '{label}':")
    print(summary)
    print()
