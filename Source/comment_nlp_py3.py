import spacy
import pandas as pd
from collections import Counter, defaultdict
import numpy as np
import re
import string
import enchant # pip install pyenchant
# from pattern.en import singularize # pip install pattern # doesnt exist in python 3

spell = enchant.Dict("en_US")


class nlpComments(object):
    def __init__(self,nlp):
        self.nlp = nlp
        self.lexicon_, self.tag_dict_ = lexi = lexi_dict() # get the dictionary

        # processing of the words in comments
        self.process_lst_ = [remove_ness_word, check_word_spelling, singularize, remove_y_word, check_word_spelling]
        self.magnitues_ = ['LOW','MED','HIGH']

    '''
    still needs a way to indicate magnitude of a compound
        ex: strong cheese flavor
        will likely use another tag google sheet
    '''

# ------------------------------------------

    def Fit_transform(self,text,add_commend_end_word = True):

        '''
        input text as a list of strings
        returns list of list of cleaned adj and noun string phrases (same as com.phrase_desc)
        creates:
            com.docs = list of spacy doc type
            com.raw_str_desc = raw text as list of strings
            com.str_desc = cleaned adj and nouns as list of strings
            com.phrase_desc = list of list of cleaned adj and noun string phrases (same as return)
        '''

        self.raw_str_desc_ = text # fit part

        # ---- transform part

        # 1--- lower case the text
        lower_str_desc = [comment.lower() for comment in text]

        # 2--- remove punctuation and replace it with a space
        no_punc_str_desc = remove_punct(lower_str_desc)

        self.clean_text_ = no_punc_str_desc
        return no_punc_str_desc

# ------------------------------------------

    def Tag_comment_list(self,text_list = None, update_progress = True):
        '''
        tags without the noun and adj filtering
        '''

        # if update progress is true default to print every 500 comments
        if update_progress == True:
            update_progress = 500

        if text_list is None:
            text_list = self.clean_text_

        full_tagged_str = []
        tagged_only_str = []
        untagged_words_list = []

        # loop over each comment in the list of comments
        for n, comment in enumerate(text_list):
            full_tagged_list, tagged_words, untagged_words = self.Tag_single_comment(comment)

            full_tagged_str.append(' '.join(full_tagged_list)) # list of strings
            tagged_only_str.append(connect_magnitude_words(tagged_words,self.magnitues_)) # list of strings
            untagged_words_list.extend(untagged_words) # list

            # for update progress
            if type(update_progress) == int:
                if n%update_progress == 0:
                    print('---> finished row {}'.format(n))


        self.tagged_full_comments_ = full_tagged_str
        self.tagged_only_comments_ = tagged_only_str
        self.untagged_words_ = untagged_words_list

        return self.tagged_only_comments_

    def Tag_single_comment(self, comment):
        '''
        comment = type str
        lexicon_compare = list of lexicon words without '_'
        lexi = lexicon dictionary
        process_lst = list of process functions

        returns = tagged_words,untagged_words
        '''
        word_list = comment.split() # split into individual words
        # add a word to the comment end to allow for the original last word to be looped over
        word_list.append('fakeword')
        tagged_words = []
        untagged_words = []
        full_tagged = []
        is_phrase = 'word1'

        #loops over each word looking at the word pair and the word before
        for i in np.arange(1,len(word_list)):

            if is_phrase == 'phrase': #skip this one
                is_phrase = 'word1'
            else:
                word1 = word_list[i-1]
                word2 = word_list[i]
                word_out, is_phrase = self.Tag_word_pair(word1,word2)

                if word_out is None: # nothing gets tagged
                    untagged_words.extend([word1])
                    full_tagged.extend([word1])

                    # untagged_words.extend([word1+' '+word2]) #saves the untagged phrase
                else:
                    tagged_words.extend([word_out])
                    full_tagged.extend([word_out])

        return full_tagged,tagged_words,untagged_words

    def Tag_word_pair(self,word1,word2):

        '''
        tags a pair of words --> raw phrase, processed phrase, raw word1, processed word2

        input word1 and word2 as type string
        returns string if phrase or word1 is tagged otherwise None
        '''

        lexicon_compare = [re.sub('-',' ',word) for word in self.lexicon_]
        lexi = self.tag_dict_

        phrase = word1 + ' ' + word2
        # check the raw phrase
        word_out = self.Check_lexi_and_keys(phrase)
        is_phrase = 'phrase' # used to identify word

        # check the processed phrase
        if word_out is None: # if no raw phrase
            processed_words = self.Vary_words_str(word1,word2)
            processed_phrase = ' '.join(processed_words)
            word_out = self.Check_lexi_and_keys(processed_phrase)
            is_phrase = 'phrase'

            # check raw word1
            if word_out is None: # if no processed phrase
                word_out = self.Check_lexi_and_keys(word1)
                is_phrase = 'word1'

                # check process word1
                if word_out is None: # if no raw word1
                    processed_word1 = processed_words[0]
                    word_out = self.Check_lexi_and_keys(processed_word1)
                    is_phrase = 'word1'

        return word_out, is_phrase


    def Check_lexi_and_keys(self,word,combiner = '/'):
        '''
        word = type str
        returns the tag of word if exists
        returns str or None
        '''
        if word in self.tag_dict_.keys():
            return combiner.join(self.tag_dict_[word])
        elif word in self.lexicon_:
            return word
        else:
            return None

    def Vary_words_str(self, word1, word2, print_words = False):
        '''
        word = type str
        self.process_lst = type list of functions
        returns list of processed words 1 & 2
        '''
        words = [word1,word2]
        out_words = []
        for process in self.process_lst_:
            inter_words = []
            for word in words:
                # process the word
                processed_word = process(word)
                # compare the processed word with the original
                better_word = self.Compare_word_porbs(word,processed_word)
                # save the word with the higher probability
                inter_words.append(better_word)
            # to see what happens at each process step
            if print_words:
                print(inter_words)
            words = [word.lower() for word in inter_words] # lower case the words in case of proper nouns
        return words

    def Compare_word_porbs(self,word1,word2):
        '''
        input is two single string words
        compares the probability of two words and returns the first word if they are very different and returns the second if they are similar
        the idea is that if the process messes up the word badly this will catch it
        returns single string
        '''
        if len(word2) > 1: # just in case word 2 doesnt exist
            p_word1 = self.nlp(word1)[0].prob
            p_word2 = self.nlp(word2)[0].prob

            if p_word1/p_word2 < .8: #.8 seems ok
                return word1
            else:
                return word2
        else: return word1


    def Make_tfidf_friendly(self):
        tfidf_friendly = []
        for comment in self.tagged_full_comments_:
            word_list = comment.split()
            word_list_ = []
            for word in word_list:
                for mag in self.magnitues_:
                    if mag in word:
                        mag_ = ' ' + mag + '_'
                        word = re.sub('/',mag_,word)
                word = re.sub('/', ' ', word)
                word = re.sub('-', '', word)
                word_list_.append(word)
            out_comment = re.sub('_', ' ' ,' '.join(word_list_))
            tfidf_friendly.append(out_comment)
        return tfidf_friendly

# ------------------------------------------
# -------------------end class--------------
# ------------------------------------------

def remove_punct(comment_list):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    return [regex.sub(' ', comment) for comment in comment_list]


def connect_magnitude_words(tagged_comment,mag_list):
    '''
    tagged_comment type string or list
    connects the magnitude words with their term and returns unique results only
    '''
    if type(tagged_comment) == str:
        in_words = tagged_comment.split()
    elif type(tagged_comment) == list:
        in_words = tagged_comment

    out_words = []

    i = 0
    while i < len(in_words):
        mag_idx = []
        words = []
        while in_words[i] in mag_list:
            # find all the mag words in a row
            words.append(in_words[i])
            if i < len(in_words)-1:
                i = i+1
            else:
                break
        words_str = '-'.join(words)
        if words != []: # append the string
            out_words.append(words_str+'_'+in_words[i])
        else:
            out_words.append(in_words[i])
        i = i+1
    return ' '.join(list(set(out_words))) # returns a string



def vary_words_lst(word1, word2, process_lst):
    '''
    !!!! didnt actually test yet Wed 4:50 21st
    word = type str
    process_lst = type list of functions
    returns list of lists processed words 1 & 2
    '''
    words = [word1, word2]
    out_words = []
    for process in process_lst:
        inter_words = []
        for word in words:
            inter_words.append(process(word))
        out_words.append(inter_words)
    return out_words


def check_word_spelling(word):
    if word != '' and spell.check(word) == False: # is it spelled wrong
        if spell.suggest(word) == []: # is there a suggestion?
            return word #no suggestion
        else:
            return spell.suggest(word)[0] #spelled wrong and a suggestion
    else:
        return word # not spelled wrong

def remove_y_word(word):
    return word.rstrip('y') #removes the y at the end of a word

def remove_ness_word(word):
    # removes -ness or -iness from a word
    if 'iness' in word:
        out_word = re.sub('iness','',word)
    else:
        out_word = re.sub('ness','',word)
    return out_word

def singularize(word):
    # vowels = {'a','i','e','o','u','y'}

    if len(word) > 1:
        if word[-1] == 's' and word[-2] != 's':
            if len(word) >= 3:
                if word[-3:] == 'ies':
                    return word[:-3]+'y'
                elif word[-2:] == 'es' and word[-3] == 'o':
                    return word[:-2]
            return word[:-1]
    return word


# ------------------------------------------

def lexi_dict():
    '''
    creates the dictionary from the google sheet output type = Defaultdict
    '''
    #reads directly from the google sheet
    df_lexi = pd.read_csv('https://docs.google.com/spreadsheets/d/1PcohvCtv_QMzwRYgR6bVXYQw3qPEm5X4fd0QWQ504PY/export?gid=0&format=csv&usp=sharing')


    # ---- this whole part keeps the two word entries together while it strips unwanted spaces
    df_lexi = df_lexi.fillna('')
    df_lexi = df_lexi.applymap(lambda x: re.sub(' ','_',x.strip()))
    lexi_list = [' '.join(list(line)).strip().split() for line in df_lexi.iloc[:,2:].values]

    # yes there is a better way to do this...
    lexi_list_ = []
    for line in lexi_list:
        line_ = []
        for word in line:
            line_.append(re.sub('_',' ',word))
        lexi_list_.append(line_)

    # ---- put a DataFrame together
    lexi = pd.DataFrame()
    lexi['lexicon'] = df_lexi.iloc[:,0]
    lexi['lexicon'] = lexi['lexicon'].apply(lambda x: re.sub('_',' ',x))
    lexi['terms'] = lexi_list_

    # create a default dict from the dataframe
    d = defaultdict(list)
    for line in range(len(lexi)):
        a = lexi.iloc[line]
        for word in a[1]:
            d[word].append(a[0])
    return lexi['lexicon'].values, d # lexicon, default dict

# ------------------------------------------
# ------------------------------------------
# ------------------------------------------

if __name__ == '__main__':
    # run -i nlp_comments2
    if not 'nlp' in locals():
        print("Loading English Module...")
        nlp = spacy.load('en')


    example_text = [ u"phenolic cloves, banana with a wheat afterthought-aaron",
     "grassy, hay, h2s, stinky socks, garlic, and pineapple",
     "smells like mom frying breakfast sausage while pancakes are cooking on the griddle: meaty and maple-y",
     "belgian yeasty isoamyl acetate, cloves, circus peanuts, phenolic, some fruit",
     "stinky sulfur, bubble gum, cloves",
     "very present maltiness that comes across as (nope, can't pinpoint it), bright citrus in retro ",
     "herbal, SO2, some orange peel, some phenolic, more herbal, more sulphur, in your face.",
     "spicy phenolic clovey-ness, white wine, cherrios, some piney herbal hops and a hint of lemon peel",
     "phenolic, rubber, shower curtain, sl fruity esters. pd  h2s, phenolic\nah",
     "lemons, sour lactic smell. strong H2S and slight SO2. phenolic. all I can smell now is the H2S.",
     "smells like summer sausage! majorly sulfury both kinds, some phenolic retronasal. sl citrus",
    u"sweet with a hint of bitter-aaron",
    "sweet with lite lingering bitter slightly sweet ",
    "slight sweet ending in slight sour, no obvious bitterness",
    "sweet and very slightly sour, mild bitterness ant the end sweet then sl sour, then really not a whole lot. ",
    "medicinal herbal bitter with some underlying sweetmenss malt sweetness, spicy then slightly bitter. ",
    "malty sweet and very sl bitter. pd ",
    "sweet\nah  mild sweetness balancd with herbal bitterness ",
    "sl sweet sl sour"]


# ------------------------------------------
# ------------------------------------------

    com = nlpComments(nlp)
    phrases = com.Fit_transform(example_text)
    '''
    input text as a list of strings
    returns list of list of cleaned adj and noun string phrases (same as com.phrase_desc)
    creates:
        com.docs = list of spacy doc type
        com.raw_str_desc = raw text as list of strings
        com.str_desc = cleaned adj and nouns as list of strings
        com.phrase_desc = list of list of cleaned adj and noun string phrases (same as return)
    '''

    # tagged_text = com.tag_words()
    '''
    returns a tagged com.str_desc (same as com.tag_str_desc)
    com.tag() creates:
        com.lexicon = list of compound words
        com.tag_dict = difaultdict of lexicon
        com.tag_str_desc = list of all words with the avaiable taged words transformed (a tagged com.str_desc)
        com.untagged Counter of all untagged words

    # need to add functionality to tag phrases
    '''
    # phrase = com.phrase_tagger()
    tagged = com.Tag_comment_list()


    # add code to move magnitude words over slashes (/) and then return TFIDF appropriate strings
    print('All Done! Type $ com. and then hit tab to see options')
    print('Run it again like this "$ run -i self.nlp_comments.py" for faster run times')

    compare = pd.DataFrame(list(zip(example_text,tagged)))
    compare.columns = ['orginal comments','doc tagged']

    from sklearn.feature_extraction.text import TfidfVectorizer
