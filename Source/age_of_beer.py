import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='.*/IPython/.*')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='pyLDAvis')
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pyLDAvis
import pyLDAvis.sklearn
from sklearn.linear_model import Lasso
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from datacleaning import Data_Loading

class Shelf_Life(object):
    def __init__(self):
        '''
        This is the init function. This just sets up the lists and dictionaries that
        are needed for the rest of the functions.
        '''
        self.lin = LinearRegression(fit_intercept=True, alpha = .001, max_iter=1000000, normalize = True, tol = .000001)
        self.lasso = Lasso(fit_intercept = True)
        n_topics = 4
        self.lda_model = LatentDirichletAllocation(n_components=n_topics, max_iter=5,
                                          learning_method='online',
                                          learning_offset=50.,
                                          random_state=0)
        self.tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                    stop_words='english')
        self.ages = {}
        self.Models = {}

        self.Beerlst = ['Beer_Abbey', 'Beer_Accumulation',
                       'Beer_AnneFrancoise', 'Beer_BenJerry', 'Beer_BlackberryBarleyWine',
                       'Beer_BotanicalIPA', 'Beer_BrettaIPA', 'Beer_CascaraQuad',
                       'Beer_CherryAlmond', 'Beer_Citradelic', 'Beer_Clutch', 'Beer_CocoaMole',
                       'Beer_Dayblazer', 'Beer_DeKoninck', 'Beer_EUROPA', 'Beer_EricsAle',
                       'Beer_FatTire', 'Beer_Frambozen', 'Beer_FrenchOakSaison',
                       'Beer_GlutenReducedPale', 'Beer_GlutinyGolden', 'Beer_GlutinyPale',
                       'Beer_Gratzer', 'Beer_Gruit', 'Beer_HeavyMelonAle',
                       'Beer_HofTenDormaal', 'Beer_Hop Tart', 'Beer_HopStout',
                       'Beer_HoppyBlonde', 'Beer_Hoptober', 'Beer_JuicyMandarina',
                       'Beer_LaFolie', 'Beer_LeTerroir', 'Beer_Lemon Ginger', 'Beer_LongTable',
                       'Beer_METIS', 'Beer_MW', 'Beer_MaryJaneIPA', 'Beer_PCS', 'Beer_PLB',
                       'Beer_PORCH SWING', 'Beer_PassionFruitDIPA', 'Beer_PearGinger',
                       'Beer_Pilsener', 'Beer_PorchSwing', 'Beer_PortagePorter',
                       'Beer_Pumpkick', 'Beer_RAPaleAle', 'Beer_Rampant', 'Beer_RyeIPA',
                       'Beer_SMH', 'Beer_SaltedBelgianStout', 'Beer_SaltedCaramel',
                       'Beer_SideTrip', 'Beer_SideTripDry', 'Beer_SideTripSemi',
                       'Beer_SkinnyDip', 'Beer_SlowRide', 'Beer_Snapshot', 'Beer_Somersault',
                       'Beer_TartLychee', 'Beer_TedsBeer', 'Beer_TransatlantiqueKriek',
                       'Beer_VRA', 'Beer_Watermelon', 'Beer_WhiskeyFatBack',
                       'Beer_WildWildDubbel', 'Beer_Yuzu']

       self.Questionlst = ['clarity','aroma','taste','body','final']

   def fit(self, df=self.df):
       '''
       This fuction creates two dictionaries that are used for predict and predict_vis
       '''
       self.df = df
       for Beer in self.Beerlst:
           temp = df[df[Beer] == 1]
            for Question in self.Questionlst:
                if not temp[temp['Question'] == Question].empty and temp[temp['Question'] == Question].shape[0] > 41:
                    Beer_Question = temp[temp['Question'] == Question]
                    unique_ages = (Beer_Question['Age'].unique())
                    y = Beer_Question.pop('Age')
                    vect = self.tf_vectorizer.fit(Beer_Question['Comments'].values.tolist())
                    tf = tf_vectorizer.transform(Beer_Question['Comments'].values.tolist())
                    LDA = self.lda_model.fit(tf)
                    lda = LDA.transform(tf)
                    Beer_Question['topic1'] = lda[:,0]
                    Beer_Question['topic2'] = lda[:,1]
                    Beer_Question['topic3'] = lda[:,2]
                    Beer_Question['topic4'] = lda[:,3]
                    Beer_Question['topic5'] = lda[:,4]
                    if len(np.unique(y_train))>1:
                        final = self.lasso.fit(Beer_Question[['Trust', 'topic1', 'topic2','topic3','topic4','topic5']], y)
                        self.models[(Beer,Question)]= (vect, LDA, final)
                        self.ages[(Beer,Question)] = unique_ages



    def predict(self, Beer, Question):
        '''
        This creates and saves a bar graph that shows how the average comment for each topic changes over time
        for a specific Beer. You can change the place that you want to save the
        '''
        thismodel = self.models[(Beer,Question)]
        new_comment = thismodel[0].transform(self.df.Comment)
        topic1, topic2, topic3, topic4, topic5 = thismodel[1].transform(new_comment)
        top_words_num = 3
        tf_feature_names = np.array(thismodel[0].get_feature_names())
        top_words = get_top_words(lda_model,tf_feature_names,top_words_num)
        all_top_words = np.array(list(set().union(*[v for v in top_words.values()])))

        for key,vals in top_words.items():
            print(key," ".join(vals))

        N = len(self.ages[(Beer,Question)])
        temp = df[df[Beer] == 1]
        Beer_Question = temp[temp['Question'] == Question]
        fig = plt.figure(figsize=(16,8))
        topic_1 = (Beer_Question[Beer,Question['Age']== age ]['topic1'].mean() for age in np.sort(self.ages[(Beer,Question)]))
        topic_2 = (Beer_Question[Beer,Question['Age']== age ]['topic2'].mean() for age in np.sort(self.ages[(Beer,Question)]))
        topic_3 = (Beer_Question[Beer,Question['Age']== age ]['topic3'].mean() for age in np.sort(self.ages[(Beer,Question)]))
        topic_4 = (Beer_Question[Beer,Question['Age']== age ]['topic4'].mean() for age in np.sort(self.ages[(Beer,Question)]))
        topic_5 = (Beer_Question[Beer,Question['Age']== age ]['topic5'].mean() for age in np.sort(self.ages[(Beer,Question)]))

        ind = np.arange(N)    # the x locations for the groups
        width = 0.35       # the width of the bars: can also be len(x) sequence

        p1 = plt.bar(ind, topic_1, width)
        p2 = plt.bar(ind, topic_2, width, bottom=topic_1)
        height_3 = (topic_1[0]+topic_2[0],topic_1[1]+topic_2[1],topic_1[2]+topic_2[2])
        p3 = plt.bar(ind, topic_3, width, bottom = height_3)
        height_4 = (height_3[0]+topic_3[0],height_3[1]+topic_3[1],height_3[2]+topic_3[2])
        p4 = plt.bar(ind, topic_4, width, bottom = height_4)
        height_5 = (height_4[0]+topic_4[0],height_4[1]+topic_4[1],height_4[2]+topic_4[2])
        p5 = plt.bar(ind, topic_5, width, bottom = height_5)

        plt.ylabel('Mean Percentage of Each Topic', fontsize = 20)
        plt.title('Mean Percentage of Each Topic that Coresponds to Month' , fontsize = 20)
        plt.xticks(ind, ('Month {}'.format(age) for age in np.sort(self.ages[(Beer,Question)])), fontsize = 20)
        plt.yticks(np.arange(0, 1, .11), fontsize = 20)
        plt.legend((p5[0], p4[0], p3[0], p2[0], p1[0]), ('Worty', 'Metallic', 'True to Brand', 'Too Bitter', 'Too Sweet'), bbox_to_anchor  = (1,.7))

        plt.savefig("../Images/{}_{}".format(Beer, Question))

    def update(self, df):
        '''
        This function updates the DataFrame that all of the functions run on and then calls Fit so that it will update the models
        '''
        self.df = pd.concact(self.df, df)
        self.df.reset_index(drop=True)
        self.fit()
        pass

    def predict_all(self):
        for Beer in self.Beerlst:
            for Question in self.Questionlst:
                self.predict(Beer, Question)

#END CLASS



def get_top_words(model, feature_names, n_top_words):
    '''
    This function is not apart of the class above. This functions
    '''
    top_words = {}
    for topic_idx, topic in enumerate(model.components_):
        _top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        top_words[str(topic_idx)] = _top_words
    return(top_words)

if __name__ == '__main__':
    DL = Data_Loading()
    if not path.exists('data.pkl'):
        doclst = DL.Document_List()
        df = DL.Create_Dataframe(doclst)
        new_comments = DL.Tokenize_Clean_Strings(df)
        DL.Clean_Beer_Names(df)
        DL.Trustworthiness(df)
        TFIDF = DL.TfidfVectorizer()
        vectorized = TFIDF.fit_transform(new_comments)
        df['Vectorized'] = vectorized
        df.to_pickle('data.pkl')
    else:
        df = pd.read_pickle('data.pkl')

    SL = Shelf_Life()
    SL.fit(df)
    SL.predict_all()
