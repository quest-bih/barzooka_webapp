"""Web app and utilities for rainbow colormap monitor
"""

import os
import os.path
import re
import math
import itertools

import flask
from flask_rq2 import RQ
from flask_mail import Mail, Message
from flask_wtf.csrf import CSRFProtect, CSRFError

from waitress import serve

from sqlalchemy import desc

import tweepy

import click
import pytest

from models import db, Biorxiv, Test
from biorxiv_scraper import find_authors, find_date, count_pages
from detect_bargraph import detect_graph_types_from_iiif
import utils

from fastai.vision import *
#ml_path = os.environ['WEBAPP_PATH']
learn = load_learner(path='/data01/webapp/barzooka/', file='export.pkl')

# Reads env file into environment, if found
_ = utils.read_env()

app = flask.Flask(__name__)
app.config['BASE_URL'] = os.environ['BASE_URL']

# For data storage
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ['SQLALCHEMY_DATABASE_URI']
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
db.init_app(app)

# For job handling
app.config['RQ_REDIS_URL'] = os.environ['RQ_REDIS_URL']
rq = RQ(app)

# For monitoring papers (until Biorxiv provides a real API)
app.config['TWITTER_APP_KEY'] = os.environ['TWITTER_APP_KEY']
app.config['TWITTER_APP_SECRET'] = os.environ['TWITTER_APP_SECRET']
app.config['TWITTER_KEY'] = os.environ['TWITTER_KEY']
app.config['TWITTER_SECRET'] = os.environ['TWITTER_SECRET']

app.config['DEBUG'] = os.environ.get('DEBUG', 0)

app.config['WEB_PASSWORD'] = os.environ['WEB_PASSWORD']

app.config['SECRET_KEY'] = os.environ['SECRET_KEY']
csrf = CSRFProtect(app)

tweepy_auth = tweepy.OAuthHandler(
    app.config['TWITTER_APP_KEY'], app.config['TWITTER_APP_SECRET'])
tweepy_auth.set_access_token(
    app.config['TWITTER_KEY'], app.config['TWITTER_SECRET'])
tweepy_api = tweepy.API(tweepy_auth)

@app.route('/')
def home():
    """Renders the website with current results
    """

    cats = flask.request.args.get('categories')
    if cats:
        cats = [int(x) for x in cats.split(',')]
    else:
        cats = [-2, -1, 0, 1, 2]

    papers = (Biorxiv.query
                     .filter(Biorxiv.parse_status.in_(cats))
                     .order_by(desc(Biorxiv.created))
                     .limit(2000)
                     .all())

    return flask.render_template('main.html', app=app, papers=papers)

@app.route('/pages/<string:paper_id>')
def pages(paper_id, prepost=1, maxshow=10):
    """Pages to show for preview
    1-index I think...
    """
    record = Biorxiv.query.filter_by(id=paper_id).first()
    if not record:
        return flask.jsonify({})

    pages = record.pages
    page_count = record.page_count

    # if requested, show all pages with each page's status
    try:
        all_pages = flask.request.args.get('all') == "1"
    except:
        all_pages = False

    show_pgs = {}
    if all_pages:
        for i in range(1, page_count + 1):
            if i in pages:
                show_pgs[i] = True
            else:
                show_pgs[i] = False
    elif pages:
        # add all detected pages up to maxshow count
        show_pgs = {i:True for i in pages[:maxshow]}
        # pad with undetected pages
        for i in pages:
            for j in range(i - prepost, min(i + prepost + 1, page_count)):
                if len(show_pgs) < maxshow:
                    if j not in pages:
                        show_pgs[j] = False
                else:
                    break
    else:
        show_pgs = {i:False for i in range(1, maxshow + 1)}

    return flask.jsonify({'pdf_url': record.pdf_url, 'pages': show_pgs})

@app.route('/detail/<string:paper_id>')
def show_details(paper_id, prepost=1, maxshow=10):
    """
    """
    record = Biorxiv.query.filter_by(id=paper_id).first()
    if not record:
        flask.flash('Sorry! Results with that ID have not been found')
        return flask.redirect('/')


    # display images
    return flask.render_template('detail.html',
        paper_id=record.id, title=record.title, url=record.url,
        pages=", ".join([str(p) for p in record.pages]),
        pages_pie=", ".join([str(p) for p in record.pages_pie]),
        pages_bardot=", ".join([str(p) for p in record.pages_bardot]),
        pages_box=", ".join([str(p) for p in record.pages_box]),
        pages_hist=", ".join([str(p) for p in record.pages_hist]),
        pages_dot=", ".join([str(p) for p in record.pages_dot]),
        pages_violin=", ".join([str(p) for p in record.pages_violin]),
        pages_positive=", ".join([str(p) for p in record.pages_positive]),
        parse_status=record.parse_status, email_sent=record.email_sent
        )

@app.route('/notify/<string:paper_id>', methods=['POST'])
@app.route('/notify/<string:paper_id>/<int:force>', methods=['POST'])
def notify_authors(paper_id, force=0):
    if flask.session.get('logged_in'):
        record = Biorxiv.query.filter_by(id=paper_id).first()
        if not record:
            return flask.jsonify(result=False, message="paper not found")

        addr = record.author_contact.values()
        addr = list(itertools.chain.from_iterable(addr))
        # don't bother everyone if there are a ton of authors
        if len(addr) > 6:
            addr = [*addr[:2], *addr[-3:]]

        if addr is [] or '@' not in "".join(addr):
            return flask.jsonify(result=False,
                message="mangled or missing email addresses")

        if force != 1 and record.email_sent == 1:
            return flask.jsonify(result=False,
                message="email already sent. use the cli to send another")

        msg = Message(
            "[JetFighter] bioRxiv manuscript {}".format(record.id),
            recipients=addr,
            reply_to=app.config['MAIL_REPLY_TO'])
        msg.body = flask.render_template("email_notification.txt",
            paper_id=paper_id,
            pages=record.pages_str,
            title=record.title,
            detail_url=flask.url_for('show_details', paper_id=paper_id))
        mail.send(msg)

        record.email_sent = 1
        db.session.merge(record)
        db.session.commit()

        return flask.jsonify(result=True, message="successfully sent")
    else:
        return flask.jsonify(result=False, message="not logged in")

@app.route('/toggle/<string:paper_id>', methods=['POST'])
def toggle_status(paper_id):
    if flask.session.get('logged_in'):
        record = Biorxiv.query.filter_by(id=paper_id).first()
        if not record:
            return flask.jsonify(result=False, message="paper not found")
        if record.parse_status > 0:
            record.parse_status = -2
        elif record.parse_status < 0:
            record.parse_status = 2
        else:
            return flask.jsonify(result=False, message="Not yet parsed")
        db.session.merge(record)
        db.session.commit()

        return flask.jsonify(result=True, message="successfully changed")
    else:
        return flask.jsonify(result=False, message="not logged in")



@app.route('/login', methods=['GET', 'POST'])
def admin_login():
    if flask.request.method == 'GET':
        if flask.session.get('logged_in'):
            flask.flash('You are already logged in!')
            return flask.redirect('/')
        return flask.render_template('login.html')

    if flask.request.form['password'] == app.config['WEB_PASSWORD']:
        flask.session['logged_in'] = True
    else:
        flask.flash('wrong password!')
        return flask.render_template('login.html')

    return flask.redirect('/')

@app.route('/logout', methods=['GET'])
def logout():
    logged_in = flask.session.get('logged_in')
    flask.session.clear()
    if logged_in:
        flask.flash("You have been successfully logged out")
    else:
        flask.flash("Not logged in! (but the session has been cleared)")
    return flask.redirect('/')

@app.errorhandler(CSRFError)
def handle_csrf_error(e):
    flask.flash('CSRF Error. Try again?')
    return flask.redirect(flask.url_for('admin_login'))



@app.cli.command()
@click.option('--count', default=500)
def retrieve_timeline(count):
    """Picks up current timeline (for testing)
    """
    # as we currently have to wait for a long time until the parsing is done
    # go through tweets twice: first only add basic info to SQL database,
    # second, parse pdf for bargraphs
    for t in tweepy.Cursor(tweepy_api.user_timeline,
            screen_name='biorxivpreprint', trim_user='True',
            include_entities=True, tweet_mode='extended').items(count):
        parse_tweet(t)


def parse_tweet(t, db=db, objclass=Biorxiv, verbose=True):
    """Parses tweets for relevant data,
       writes each paper to the database,
       dispatches a processing job to the processing queue (rq)
    """
    try:
        text = t.extended_tweet["full_text"]
    except AttributeError:
        pass

    try:
        text = t.full_text
    except AttributeError:
        text = t.text

    if verbose:
        print(t.id_str, text[:25], end='\r')
    if not db:
        return

    try:
        url = t.entities['urls'][0]['expanded_url']
        code = os.path.basename(url)
    except:
        print('Error parsing url/code from tweet_id', t.id_str)
        return

    try:
        title = re.findall('(.*?)\shttp', text)[0]
    except:
        # keep ASCII only (happens with some Test tweets)
        title = re.sub(r'[^\x00-\x7f]', r'', text)

    obj = objclass(
        id=code,
        created=t.created_at,
        title=title,
    )

    obj = db.session.merge(obj)
    db.session.commit()

    # Only add to queue if not yet processed and if we actually want to do all the processing
    if obj.parse_status == 0:
        process_paper.queue(obj)


@rq.job(timeout='30m')
def process_paper(obj):
    #Processes paper starting from url/code
    #
    #1. get object, find page count and posted date
    #2. detect graph classes
    #3. if bargraph, get authors
    #4. update database entry with graph detection and author info
    with app.app_context():
        obj = db.session.merge(obj)

        if obj.page_count == 0:
            obj.page_count = count_pages(obj.id)

        if obj.posted_date == "":
            obj.posted_date = find_date(obj.id)

        class_pages = detect_graph_types_from_iiif(obj.id, obj.page_count, learn)

        obj.pages = class_pages["bar"]
        obj.pages_pie = class_pages["pie"]
        obj.pages_hist = class_pages["hist"]
        obj.pages_bardot = class_pages["bardot"]
        obj.pages_box = class_pages["box"]
        obj.pages_dot = class_pages["dot"]
        obj.pages_violin = class_pages["violin"]
        obj.pages_positive = class_pages["positive"]


        if (len(obj.pages) + len(obj.pages_pie)) > 0:
           obj.parse_status = 1
           if obj.author_contact is None:
                obj.author_contact = find_authors(obj.id)
        else:
            obj.parse_status = -1

        db.session.merge(obj)
        db.session.commit()


## NOTE: NEEDS WORK
@pytest.fixture()
def test_setup_cleanup():
    # should only be one, but... just in case
    for obj in Test.query.filter_by(id='172627v1').all():
        db.session.delete(obj)
    db.session.commit()

    # Delete temporary row
    for obj in Test.query.filter_by(id='172627v1').all():
        db.session.delete(obj)
    db.session.commit()

def test_integration(test_setup_cleanup):
    """Submit job for known jet colormap. Remove from database beforehand.
    Write to database.
    Check for written authors.
    """

    #testq = rq.Queue('testq', is_async=False)

    preobj = Test(id='172627v1')
    testq.enqueue(process_paper, preobj)

    postobj = Test.query.filter_by(id='172627v1').first()

    # check that document was correctly identified as having a rainbow colormap
    assert postobj.parse_status

    # check that authors were correctly retrieved
    authors = postobj.author_contact
    assert authors['corr'] == ['t.ellis@imperial.ac.uk']
    assert set(authors['all']) == set([
        'o.borkowski@imperial.ac.uk', 'carlos.bricio@gmail.com',
        'g.stan@imperial.ac.uk', 't.ellis@imperial.ac.uk'])

if __name__ == "__main__":
    #app.run(debug=True, threaded=True, use_reloader=True)
    serve(app, host='127.0.0.1', port=5000, url_scheme='https')