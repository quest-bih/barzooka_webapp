"""Take a biorxiv identifier and resolves the link to the pdf
"""

import flask

from waitress import serve

import requests
from bs4 import BeautifulSoup

app = flask.Flask(__name__)

def test_parse_page():
    pass

def parse_page(html):
    soup = BeautifulSoup(html, "lxml")

    a = soup.find_all("a", {"class": "article-dl-pdf-link"})[0]
    url = a.attrs['href']
    
    return 'https://www.biorxiv.org' + url

#e.g. localhost:8300/biorxiv/815209v1
@app.route('/<string:source>/<string:paper_id>')
def pages(source, paper_id):
    """Take the paper_id and returns the pdf url
    """
    # only know how to resolve biorxiv pdf's right now
    if source != 'biorxiv':
        return flask.abort(400)

    #paper_id = paper_id.split('.', 1)[0]
    url = "https://www.biorxiv.org/content/10.1101/{}".format(paper_id)  
    r = requests.get(url)

    try:
        url = parse_page(r.text)
        return flask.redirect(url)
    except:
        return flask.abort(400)

if __name__ == "__main__":
	app.run(debug=True, threaded=True, use_reloader=False, host='127.0.0.1', port=8300)
    #serve(app, host='127.0.0.1', port=8300) 
