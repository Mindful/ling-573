import unittest
import datetime
from collections import namedtuple
from spacy.tokens import Doc, Span
from preprocessing.topic_doc_group import DocumentGroup, DocGroupArticle, clean_text
from data import Topic

Metadata = namedtuple('Metadata', ['id', 'title', 'narrative'])
Article = namedtuple('Article', ['id', 'date', 'type', 'headline', 'paragraphs'])

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        topic_metadata = Metadata(id='D1001A', title='Columbine Massacre', narrative='Here is a sample narrative')
        article_1 = Article(id='APW19990421.0284', date=datetime.date(1999, 4, 21), type=None, headline=None, paragraphs=["LITTLETON, Colo. (AP) -- The sheriff's initial estimate of as  \nmany as 25 dead in the Columbine High massacre was off the mark \napparently because the six SWAT teams that swept the building \ncounted some victims more than once.", 'Sheriff John Stone said Tuesday afternoon that there could be as  \nmany as 25 dead. By early Wednesday, his deputies said the death \ntoll was 15, including the two gunmen.', "The discrepancy occurred because the SWAT teams that picked  \ntheir way past bombs and bodies in an effort to secure building \ncovered overlapping areas, said sheriff's spokesman Steve Davis.", "``There were so many different SWAT teams in there, we were  \nconstantly getting different counts,'' Davis said.", 'As they gave periodic updates through the night, Davis and Stone  \nemphasized the death toll was unconfirmed. They said their priority \nwas making sure the school was safe.'])
        article_2 = Article(id='APW19990422.0095', date=datetime.date(1999, 4, 22), type=None, headline=None, paragraphs=['BURBANK, Calif. (AP) -- Republican presidential candidate Pat  \nBuchanan says stricter gun laws could not have prevented the deadly \nschool shootings in Littleton, Colo.', "``The question is who has the weapons, the good people or those  \nwho are ugly and warped,'' he said. ``The problem began long before \nthey walked into school.''", 'Fifteen people, including the two killers, died Tuesday in a  \nshooting and bombing spree at Columbine High School.', "``The massacre is a tragic reflection of the dark side of  \nAmerican society,'' Buchanan told reporters Wednesday. ``At \nLittleton, America got a glimpse of the last stop on that train to \nhell America boarded decades ago when we declared that God is dead \nand that each of us is his or her own god who can make up the rules \nas we go along.''"])
        article_3 = Article(id='APW19990427.0078', date=datetime.date(1999, 4, 27), type=None, headline=None, paragraphs=['LITTLETON, Colo. (AP) -- The day that Columbine High School  \nstudents are to return to class has been delayed because so many \nhave been attending funerals for students killed in the April 20 \nmassacre, an administrator said Tuesday.', 'The students are scheduled to begin classes Monday at another  \nschool a few miles away in afternoon sessions until the end of the \nyear, said Barbara Monseu, area administrator for the Jefferson \nCounty School District.', 'Students were originally scheduled to go back Thursday.', "Calling the return to the classroom ``a very important next step  \nin the process of healing,'' Monseu said the first thing Columbine \nstudents will do when they arrive at Chatfield High School is \nattend an assembly to reunite them with their teachers.", "``The teachers are very anxious to see their students again,''  \nshe said.", 'Columbine and Chatfield are sports rivals, but junior John Danos  \nsaid he welcomed the newcomers.', "``I'm fine with it,'' he said. ``We're going to treat them  \nnormal.''", 'To accommodate the 1,965 Columbine students, the school day is  \nbeing split with Chatfield students beginning early in the day and \nColumbine students showing up shortly before 1 p.m.', "To prepare for the resumption of classes, some of Columbine's  \n158 teachers met Tuesday morning for the first time since two \nstudent gunmen fatally shot 12 fellow students and a teacher before \ntaking their own lives."])
        
        self.topic = Topic(topic_metadata, [article_1, article_2, article_3])


    def test_create_document_group(self):
        document_group = DocumentGroup(self.topic)
        
        self.assertEqual(document_group.topic_id, self.topic.id)
        self.assertEqual(document_group.title, self.topic.title)
        self.assertEqual(type(document_group.narrative), Doc)
        self.assertEqual(document_group.narrative.text, self.topic.narrative)
        self.assertEqual(len(document_group.articles), 3)
        self.assertEqual(type(document_group.articles[0]), DocGroupArticle)


    def test_create_doc_group_article(self):
        document_group = DocumentGroup(self.topic)
        article_1 = document_group.articles[0]

        self.assertEqual(article_1.id, 'APW19990421.0284')
        self.assertEqual(article_1.date, datetime.date(1999, 4, 21))
        self.assertEqual(article_1.type, None)
        self.assertEqual(article_1.headline, None)
        self.assertIs(type(article_1.paragraphs[0]), Doc)


    def test_create_doc_group_article_empty_paragraphs(self):
        paragraphs_with_empty = [
                                    'The Nixon administration did it. So did Oliver North. And so does Carole Harris.',
                                    'Shred, that is.',
                                    '',
                                    'Every day during the summer, Ms. Harris, 65, opens her mail on the sun porch of her summer cottage in Buchanan, N.Y.', 
                                    ''
                                ]
        article = Article(id='APW19990421.0284', 
                          date=datetime.date(1999, 4, 21), 
                          type=None, 
                          headline=None, 
                          paragraphs=paragraphs_with_empty)
    
        doc_group_article = DocGroupArticle(article)

        self.assertEqual(len(doc_group_article.paragraphs), 3)


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
        self.assertEqual(len(paragraph_1_sents), 1)
        self.assertIs(type(paragraph_1_sents[0]), Span)
        self.assertNotEqual(paragraph_1_sents[0].text, 'LITTLETON, Colo. (AP) --')
        self.assertEqual(paragraph_1_sents[0].text, "The sheriff's initial estimate of as many as 25 dead in the Columbine High massacre was off the mark apparently because the six SWAT teams that swept the building counted some victims more than once.")

        # ents
        self.assertEqual(type(paragraph_1_ents[0]), Span)
        self.assertEqual([ent.text for ent in paragraph_1_ents], ["as many as 25", "Columbine High", "six", "SWAT"])

        # noun_chucks
        self.assertEqual(type(paragraph_1_noun_chunks[0]), Span)
        self.assertEqual([chunk.text for chunk in paragraph_1_noun_chunks], ["The sheriff's initial estimate", "the Columbine High massacre", "the mark", "the six SWAT teams", "the building", "some victims"])

        # cosine vector similarity comparisons
        self.assertEqual(article_1_paragraph_1.text, "The sheriff's initial estimate of as many as 25 dead in the Columbine High massacre was off the mark apparently because the six SWAT teams that swept the building counted some victims more than once.")
        self.assertEqual(article_2_paragraph_1.text, "Republican presidential candidate Pat Buchanan says stricter gun laws could not have prevented the deadly school shootings in Littleton, Colo.")
        self.assertEqual(article_1_paragraph_2.text, "The discrepancy occurred because the SWAT teams that picked their way past bombs and bodies in an effort to secure building covered overlapping areas, said sheriff's spokesman Steve Davis.")
       
        paragraph_sim_1 = article_1_paragraph_1.similarity(article_2_paragraph_1)
        self.assertEqual(paragraph_sim_1, 0.7906798944810245)

        paragraph_sim_2 = article_1_paragraph_1.similarity(article_1_paragraph_2)
        self.assertEqual(paragraph_sim_2, 0.9508306138290268)


    def test_cleaning_text_removes_AP_beginning(self):
        text = "LITTLETON, Colo. (AP) -- The sheriff's initial estimate of as many as 25 dead in the Columbine High."
        cleaned = clean_text(text)
        self.assertEqual(cleaned, "The sheriff's initial estimate of as many as 25 dead in the Columbine High.")


    def test_cleaning_text_removes_spurios_newline_chars(self):
        text = "The sheriff's initial estimate of as  \nmany as 25 dead in the Columbine High."
        cleaned = clean_text(text)
        self.assertEqual(cleaned, "The sheriff's initial estimate of as many as 25 dead in the Columbine High.")


    def test_cleaning_text_removes_spurios_newline_chars_touching_words(self):
        text = "The New York Times plans two pages of stories, photos and\ngraphics on the aftermath of the school shooting in a Denver suburb\nthat left 15 dead."
        cleaned = clean_text(text)
        self.assertEqual(cleaned, "The New York Times plans two pages of stories, photos and graphics on the aftermath of the school shooting in a Denver suburb that left 15 dead.")


    def test_cleaning_text_removes_spurios_newline_chars_touching_words_2(self):
        text = "Pastors in Jonesboro, Ark.,\nscene of an earlier school shooting"
        cleaned = clean_text(text)
        self.assertEqual(cleaned, "Pastors in Jonesboro, Ark., scene of an earlier school shooting")


    def test_cleaning_text_removes_location(self):
        text = "NEW YORK _ The Rev. Al Sharpton stepped to the microphone outside the Bronx County Couthouse and bellowed"
        cleaned = clean_text(text)
        self.assertEqual(cleaned, "The Rev. Al Sharpton stepped to the microphone outside the Bronx County Couthouse and bellowed")


    def test_cleaning_text_removes_location_2(self):
        text = "BAGHDAD, Iraq (AP) The Rev. Al Sharpton stepped to the microphone outside the Bronx County Couthouse and bellowed"
        cleaned = clean_text(text)
        self.assertEqual(cleaned, "The Rev. Al Sharpton stepped to the microphone outside the Bronx County Couthouse and bellowed")


    def test_cleaning_text_removes_location_3(self):
        text = "KHARTOUM, Sudan _ The Rev. Al Sharpton stepped to the microphone outside the Bronx County Couthouse and bellowed"
        cleaned = clean_text(text)
        self.assertEqual(cleaned, "The Rev. Al Sharpton stepped to the microphone outside the Bronx County Couthouse and bellowed")


    def test_cleaning_text_removes_link(self):
        text = "On the Net:"
        cleaned = clean_text(text)
        self.assertEqual(cleaned, "")


    def test_cleaning_text_removes_link_2(self):
        # not certain what would actually be desired here. delete the whole rest of the sentence? need to make sure it won't break the text in other use cases, if so
        text = "http://ifats.org is the International Fat Applied Technology Society"
        cleaned = clean_text(text)
        self.assertEqual(cleaned, "is the International Fat Applied Technology Society")


    def test_cleaning_text_removes_email_line(self):
        text = "E-mail: triggp(at)nytimes.com.'"
        cleaned = clean_text(text)
        self.assertEqual(cleaned, "")


    def test_cleaning_text_removes_story_filed_by(self):
        text = "Story Filed By Cox Newspapers"
        cleaned = clean_text(text)
        self.assertEqual(cleaned, "")


    def test_cleaning_text_removes_for_use_by(self):
        text = "For Use By Clients of the New York Times News Service"
        cleaned = clean_text(text)
        self.assertEqual(cleaned, "")

    def test_cleaning_text_removes_phone(self):
        text = "Phone: (888) 603-1036"
        cleaned = clean_text(text)
        self.assertEqual(cleaned, "")


    def test_cleaning_text_removes_pager(self):
        text = "Pager: (800) 946-4645 (PIN 599-4539)."
        cleaned = clean_text(text)
        self.assertEqual(cleaned, "")


    def test_cleaning_text_removes_technical_problems(self):
        text = "TECHNICAL PROBLEMS:   Peter Trigg"
        cleaned = clean_text(text)
        self.assertEqual(cleaned, "")


    def test_cleaning_text_removes_photos(self):
        text = "PHOTOS AND GRAPHICS:"
        cleaned = clean_text(text)
        self.assertEqual(cleaned, "")


if __name__ == '__main__':
    unittest.main()
