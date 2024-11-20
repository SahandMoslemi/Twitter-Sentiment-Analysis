import os
import pandas as pd
import matplotlib.pyplot as plt

# Read the dataset
training_df = (
    pd.read_csv(os.path.join('data', 'twitter_training.csv'))
    .rename(columns={'Positive': 'sentiment', 
                     'im getting on borderlands and i will murder you all ,': 'text'})
    .sample(n=500, random_state=2)
)

# Calculate the ratio of sentiment labels
label_counts = training_df['sentiment'].value_counts(normalize=True)

# Plot and save the pie chart
plt.figure(figsize=(8, 8))
label_counts.plot.pie(autopct='%1.1f%%', startangle=90, title="Sentiment Ratios")
plt.ylabel("")  # Remove y-axis label
plt.savefig("sentiment_ratios.png")
plt.close()

# Calculate the average number of words in the 'text' column
training_df['WordCount'] = training_df['text'].apply(lambda x: len(str(x).split()))
average_word_count = training_df['WordCount'].mean()

# Plot and save the bar chart for average word count
plt.figure(figsize=(8, 6))
plt.bar(["Average Word Count"], [average_word_count])
plt.title("Average Number of Words in Text")
plt.ylabel("Words")
plt.savefig("average_word_count.png")
plt.close()

print("Plots saved successfully as 'sentiment_ratios.png' and 'average_word_count.png'.")
