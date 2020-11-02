import matplotlib.pyplot as plt
import wordcloud as wc

from src.statistics.abstract_plotter import AbstractPlotter

stopwords = ["a", "an", "is", "it", "the", "does", "do", "are", "you", "that",
             "they", "doe", "this", "there", "hi", "his", "her", "its", "picture", "can", "he", "she", "bu", "us",
             "photo"]
stopwords = []


class WordCloud(AbstractPlotter):
    def __init__(self, path, questions, dataset, suffix, stopwords=[]):
        super(WordCloud, self).__init__(path, self.__class__.__name__, suffix)

        # questions = dataset.target_questions.t().numpy()
        questions_decoded = " ".join(
            dataset.question_vocab.decode(x, stop_at_end=True, ignored=['<SOS>', '<PAD>']) for x in questions)

        # take relative word frequencies into account, lower max_font_size
        wordcloud = wc.WordCloud(background_color="white", max_font_size=40, max_words=80,
                                 stopwords=stopwords, prefer_horizontal=1, width=400, height=200) \
            .generate(questions_decoded)

        plt.figure()
        plt.imshow(wordcloud)
        plt.axis("off")
