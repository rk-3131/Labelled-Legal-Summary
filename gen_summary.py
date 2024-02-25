from gensim.summarization import summarize

# Load your text from a file or provide it as a string
with open("1953_L_1.txt", "r") as file:
    lines = file.readlines()

# Process the data to separate sentences and labels
sentences = [line.split('\t')[0] for line in lines]
labels = [line.split('\t')[1].strip() for line in lines]

# Create a dictionary to store sentences based on their labels
sentences_by_label = {}
for sentence, label in zip(sentences, labels):
    if label not in sentences_by_label:
        sentences_by_label[label] = []
    sentences_by_label[label].append(sentence)

# Specify the ratio of the original text length to be included in the summary
summary_ratio = 0.2  # You can adjust this value as needed

# Generate and print summaries based on labels
for label, sentences in sentences_by_label.items():
    text = ' '.join(sentences)
    summary = summarize(text, ratio=summary_ratio)
    print(f"Summary for label '{label}':")
    print(summary)
    print()
