# Tweet Clustering Using SBERT

This project demonstrates how to retrieve tweets from Twitter and cluster them into distinct groups based on their semantic content using Sentence-BERT (SBERT). The approach involves fetching tweets via the Twitter API, encoding them into numerical representations with SBERT, and applying clustering algorithms to group similar tweets together.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

## MacOS Installation instructions
Download Github desktop https://desktop.github.com/download/

After opening the app it will ask you to login to your guthub account or create a free account.

After creating or loggin in sign in to this app.

Press clone repository from internet, URL and paste this link https://github.com/gr13nka/SBert-Sociology.git 

Press clone button and it will download all files.

Download Anaconda to run code https://www.anaconda.com/download/success

We need a regular one not a miniconda, it will download a .pkg file, open it and follow instructions.

After install open anaconda navigator and press on JupyterLab Launch button.

After load it will open a tab in your browser.

On the left there will a file browser, navigate to where the project fikes are stored, by default they are in Documents/Github/Sbert-Sociology and open the main.ipynb

The code will appear, press the play button on the top or click on code and press shift enter to run.

These windows with code and text are called cells, while the cell is working it will show a [*] icon, after it finishes it will show a number like this [1]

run the first 2 cellsIt will download neccesary code for project.

Then Run the main big code it will show you the graphs and open an interactive cluster analysis tool.


## Introduction

Understanding patterns and trends in social media conversations can provide valuable insights into public opinion and emerging topics. This project leverages natural language processing techniques to analyze tweets by:

1. **Fetching Tweets**: Utilizing the Twitter API to collect tweets based on specified keywords or hashtags.
2. **Encoding Tweets**: Transforming the textual data into numerical embeddings using SBERT, which captures the semantic essence of the text.
3. **Clustering**: Grouping the encoded tweets into clusters that represent distinct topics or themes.

## Features

- **Tweet Retrieval**: Collect tweets in real-time based on search queries.
- **Semantic Encoding**: Use SBERT to generate meaningful embeddings for each tweet.
- **Clustering**: Apply clustering algorithms like K-Means to organize tweets into coherent groups.
- **Visualization**: (Optional) Visualize the clustered data to interpret the results effectively.

## Installation

To set up the project environment, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/tweet-clustering-sbert.git
   cd tweet-clustering-sbert
   ```

2. **Create a Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Required Packages**:
   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` file includes:
   - `tweepy`: For accessing the Twitter API.
   - `sentence-transformers`: For generating sentence embeddings using SBERT.
   - `scikit-learn`: For clustering algorithms and evaluation.

## Configuration

To access the Twitter API, you need to set up authentication credentials:

1. **Create a Twitter Developer Account**: Register at the [Twitter Developer Portal](https://developer.twitter.com/en/apply-for-access) and create a new application to obtain your API keys and access tokens.

2. **Configure Authentication**: In the project directory, create a file named `config.py` with the following content:

   ```python
   consumer_key = 'your_consumer_key'
   consumer_secret = 'your_consumer_secret'
   access_token = 'your_access_token'
   access_token_secret = 'your_access_token_secret'
   ```

   Replace the placeholders with your actual Twitter API credentials.

## Usage

With the environment set up and configured, you can run the main script to fetch and cluster tweets:

1. **Run the Script**:
   ```bash
   python main.py
   ```

2. **Analyze the Output**: The script will output the clustered tweets, displaying sample tweets from each cluster. You can modify the script to save results to a file or database as needed.

## Contributing

Contributions to this project are welcome. If you have suggestions for improvements or encounter issues, please open an issue or submit a pull request on the [GitHub repository](https://github.com/yourusername/tweet-clustering-sbert).

## References

- **Tweepy Documentation**: [https://pypi.org/project/tweepy/](https://pypi.org/project/tweepy/)
- **Sentence Transformers Documentation**: [https://sbert.net/](https://sbert.net/)
- **Scikit-learn Documentation**: [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)
- **Twitter Developer Portal**: [https://developer.twitter.com/en/apply-for-access](https://developer.twitter.com/en/apply-for-access)

By following this guide, you can set up and run the project to analyze and cluster tweets, gaining insights into various sociological phenomena reflected in social media conversations. 
