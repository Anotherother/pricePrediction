import requests
from pyquery import PyQuery
import pandas as pd

def crawl_bitcoin(no_of_last_pages=370):
    print ('parsing bitcoin')
    headers = { 'Accept-Encoding': 'identity' }

    articles = []

    for index in range(1, no_of_last_pages):
        r = requests.get("https://news.bitcoin.com/page/" + str(index), headers=headers)

        pq = PyQuery(r.text)
        postTags = pq('div.td_module_wrap')

        print ('Crawled page ' + str(index) + ': Extracted ' + str(len(postTags)))
        for postTag in postTags:
            postTagObj = PyQuery(postTag)
            time =  postTagObj('time').attr('datetime')
            url =  postTagObj('div > h3 > a').attr('href')
            title = postTagObj('div > h3 > a').text()
            article = { "url": url, "title": title.encode("utf-8")  , "time": time }
            articles.append(article)

        print ('news.bitcoin.com: ' + str(len(articles)) + ' articles has been extracted.')

    articles = pd.DataFrame(articles)
    articles.to_csv('./data/news_bitcoin_com.csv', encoding='utf-8')
    return articles

def crawl_coindesk():
    headers = {'Accept-Encoding': 'identity'}
    r = requests.get('http://www.coindesk.com/category/news/', headers=headers)

    articles = []

    pq = PyQuery(r.text)
    post_tags = pq('div.post')

    for postTag in post_tags:
        postTagObj = PyQuery(postTag)
        time = postTagObj('time').attr('datetime')
        url = postTagObj('div > a').attr('href')
        title = postTagObj('div > a').attr('title')
        article = {'url': url, 'title': title, 'time': time}
        articles.append(article)

    articles = pd.DataFrame(articles)
    articles.to_csv('./data/coindesk.csv', encoding='utf-8')
    return articles


def crawl_cryptocoinnews(section, no_of_last_pages):
    print('parsing cryptocoinnews.com/' + section)
    headers = {'Accept-Encoding': 'identity'}

    no_of_news = 0
    article_pandas = pd.DataFrame({'url': [], 'title': [], 'time': []})
    for index in range(1, no_of_last_pages):

        r = requests.get("https://www.cryptocoinsnews.com/" + section + "/page/" + str(index), headers=headers)

        pq = PyQuery(r.text)
        post_wrapper_tag = 'type-post'
        postTags = pq('div.' + post_wrapper_tag)

        print('parse page ' + str(index) + ': Extracted ' + str(len(postTags)))

        for postTag in postTags:
            postTagObj = PyQuery(postTag)

            splitted_date = postTagObj('span.date').text().split('/')

            time = str(splitted_date[2]) + '-' + str(splitted_date[1]) + '-' + str(splitted_date[0])\

            url = postTagObj('div > h3 > a').attr('href')
            title = postTagObj('div > h3 > a').text()
            article = {"url": url, "title": title.encode('utf-8'), "time": time}

            no_of_news = no_of_news + 1
            article_pandas = article_pandas.append(pd.DataFrame(data = [article], columns=['url', 'title', 'time']))

        print('cryptocoinnews.com: ' + str(no_of_news) + ' articles has been extracted.')

    article_pandas.to_csv('./data/cryptocoinnews.csv', encoding='utf-8')
    return article_pandas