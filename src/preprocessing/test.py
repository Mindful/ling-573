import unittest
import datetime
from collections import namedtuple
from spacy.tokens import Doc, Span
from preprocessing.topic_doc_group import DocumentGroup, DocGroupArticle, clean_text
from data import Topic


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        Metadata = namedtuple('Metadata', ['id', 'title', 'narrative'])
        topic_metadata = Metadata(id='D1001A', title='Columbine Massacre', narrative='')

        Article = namedtuple('Article', ['id', 'date', 'type', 'headline', 'paragraphs'])
        article_1 = Article(id='APW19990421.0284', date=datetime.date(1999, 4, 21), type=None, headline=None, paragraphs=["LITTLETON, Colo. (AP) -- The sheriff's initial estimate of as  \nmany as 25 dead in the Columbine High massacre was off the mark \napparently because the six SWAT teams that swept the building \ncounted some victims more than once.", 'Sheriff John Stone said Tuesday afternoon that there could be as  \nmany as 25 dead. By early Wednesday, his deputies said the death \ntoll was 15, including the two gunmen.', "The discrepancy occurred because the SWAT teams that picked  \ntheir way past bombs and bodies in an effort to secure building \ncovered overlapping areas, said sheriff's spokesman Steve Davis.", "``There were so many different SWAT teams in there, we were  \nconstantly getting different counts,'' Davis said.", 'As they gave periodic updates through the night, Davis and Stone  \nemphasized the death toll was unconfirmed. They said their priority \nwas making sure the school was safe.'])
        article_2 = Article(id='APW19990422.0095', date=datetime.date(1999, 4, 22), type=None, headline=None, paragraphs=['BURBANK, Calif. (AP) -- Republican presidential candidate Pat  \nBuchanan says stricter gun laws could not have prevented the deadly \nschool shootings in Littleton, Colo.', "``The question is who has the weapons, the good people or those  \nwho are ugly and warped,'' he said. ``The problem began long before \nthey walked into school.''", 'Fifteen people, including the two killers, died Tuesday in a  \nshooting and bombing spree at Columbine High School.', "``The massacre is a tragic reflection of the dark side of  \nAmerican society,'' Buchanan told reporters Wednesday. ``At \nLittleton, America got a glimpse of the last stop on that train to \nhell America boarded decades ago when we declared that God is dead \nand that each of us is his or her own god who can make up the rules \nas we go along.''"])
        article_3 = Article(id='APW19990427.0078', date=datetime.date(1999, 4, 27), type=None, headline=None, paragraphs=['LITTLETON, Colo. (AP) -- The day that Columbine High School  \nstudents are to return to class has been delayed because so many \nhave been attending funerals for students killed in the April 20 \nmassacre, an administrator said Tuesday.', 'The students are scheduled to begin classes Monday at another  \nschool a few miles away in afternoon sessions until the end of the \nyear, said Barbara Monseu, area administrator for the Jefferson \nCounty School District.', 'Students were originally scheduled to go back Thursday.', "Calling the return to the classroom ``a very important next step  \nin the process of healing,'' Monseu said the first thing Columbine \nstudents will do when they arrive at Chatfield High School is \nattend an assembly to reunite them with their teachers.", "``The teachers are very anxious to see their students again,''  \nshe said.", 'Columbine and Chatfield are sports rivals, but junior John Danos  \nsaid he welcomed the newcomers.', "``I'm fine with it,'' he said. ``We're going to treat them  \nnormal.''", 'To accommodate the 1,965 Columbine students, the school day is  \nbeing split with Chatfield students beginning early in the day and \nColumbine students showing up shortly before 1 p.m.', "To prepare for the resumption of classes, some of Columbine's  \n158 teachers met Tuesday morning for the first time since two \nstudent gunmen fatally shot 12 fellow students and a teacher before \ntaking their own lives."])
        
        self.topic = Topic(topic_metadata, [article_1, article_2, article_3])


    def test_create_document_group(self):
        document_group = DocumentGroup(self.topic)
        
        self.assertEqual(document_group.topic_id, self.topic.id)
        self.assertEqual(document_group.title, self.topic.title)
        self.assertEqual(document_group.narrative, self.topic.narrative)
        self.assertEqual(len(document_group.articles), 3)
        self.assertEqual(type(document_group.articles[0]), DocGroupArticle)


    def test_create_doc_group_article(self):
        document_group = DocumentGroup(self.topic)
        article_1 = document_group.articles[0]

        self.assertEqual(article_1.id, 'APW19990421.0284')
        self.assertEqual(article_1.date, datetime.date(1999, 4, 21))
        self.assertEqual(article_1.type, None)
        self.assertEqual(article_1.headline, None)
        self.assertEqual(len(article_1.unprocessed_paragraphs), len(article_1.paragraphs))
        self.assertIs(type(article_1.unprocessed_paragraphs[0]), str)
        self.assertIs(type(article_1.paragraphs[0]), Doc)


    def test_doc_group_article_spacy_stuff(self):
        document_group = DocumentGroup(self.topic)
        article_1 = document_group.articles[0]
        article_2 = document_group.articles[1]

        article_1_paragraph_1 = article_1.paragraphs[0]
        article_1_paragraph_2 = article_1.paragraphs[2]
        article_2_paragraph_1 = article_2.paragraphs[0]

        paragraph_1_sents = list(article_1_paragraph_1.sents)
        paragraph_1_ents = list(article_1_paragraph_1.ents)
        paragraph_1_noun_chunks = list(article_1_paragraph_1.noun_chunks)

        # sentences
        self.assertEqual(len(paragraph_1_sents), 2)
        self.assertIs(type(paragraph_1_sents[0]), Span)
        self.assertEqual(paragraph_1_sents[0].text, 'LITTLETON, Colo. (AP) --')
        self.assertEqual(paragraph_1_sents[1].text, "The sheriff's initial estimate of as many as 25 dead in the Columbine High massacre was off the mark apparently because the six SWAT teams that swept the building counted some victims more than once.")

        # ents
        self.assertEqual(type(paragraph_1_ents[0]), Span)
        self.assertEqual([ent.text for ent in paragraph_1_ents], ["LITTLETON", "Colo.", "AP", "as many as 25", "Columbine High", "six", "SWAT"])

        # noun_chucks
        self.assertEqual(type(paragraph_1_noun_chunks[0]), Span)
        self.assertEqual([chunk.text for chunk in paragraph_1_noun_chunks], ["LITTLETON", "Colo. (AP", "The sheriff's initial estimate", "the Columbine High massacre", "the mark", "the six SWAT teams", "the building", "some victims"])

        # cosine vector similarity comparisons
        self.assertEqual(article_1_paragraph_1.text, "LITTLETON, Colo. (AP) -- The sheriff's initial estimate of as many as 25 dead in the Columbine High massacre was off the mark apparently because the six SWAT teams that swept the building counted some victims more than once.")
        self.assertEqual(article_2_paragraph_1.text, "BURBANK, Calif. (AP) -- Republican presidential candidate Pat Buchanan says stricter gun laws could not have prevented the deadly school shootings in Littleton, Colo.")
        self.assertEqual(article_1_paragraph_2.text, "The discrepancy occurred because the SWAT teams that picked their way past bombs and bodies in an effort to secure building covered overlapping areas, said sheriff's spokesman Steve Davis.")
       
        paragraph_sim_1 = article_1_paragraph_1.similarity(article_2_paragraph_1)
        self.assertEqual(paragraph_sim_1, 0.819380660099558)

        paragraph_sim_2 = article_1_paragraph_1.similarity(article_1_paragraph_2)
        self.assertEqual(paragraph_sim_2, 0.939751888684158)


    def test_cleaning_text_removes_spurios_newline_chars(self):
        text = "LITTLETON, Colo. (AP) -- The sheriff's initial estimate of as  \nmany as 25 dead in the Columbine High."
        cleaned = clean_text(text)

        self.assertEqual(cleaned, "LITTLETON, Colo. (AP) -- The sheriff's initial estimate of as many as 25 dead in the Columbine High.")



if __name__ == '__main__':
    unittest.main()
