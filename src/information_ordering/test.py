import unittest
from collections import namedtuple
import datetime
from data import Topic
from information_ordering.ordering import Ordering
from preprocessing.topic_doc_group import DocumentGroup, DocGroupArticle, clean_text
from content_selection.selection import Content
from common import NLP

Metadata = namedtuple('Metadata', ['id', 'title', 'narrative'])
Article = namedtuple('Article', ['id', 'date', 'type', 'headline', 'paragraphs'])
Realization = namedtuple('Realization', ['selected_content', 'doc_group', 'realized_content'])

class TestOrdering(unittest.TestCase):
    def setUp(self):
        topic_metadata = Metadata(id='D1001A', title='Columbine Massacre', narrative='Here is a sample narrative')
        article_1 = Article(id='APW19990421.0284', date=datetime.date(1999, 4, 21), type=None, headline=None, 
                            paragraphs=["LITTLETON, Colo. (AP) -- The sheriff's initial estimate of as  \nmany as 25 dead in the Columbine High massacre was off the mark \napparently because the six SWAT teams that swept the building \ncounted some victims more than once.",
                                        'Sheriff John Stone said Tuesday afternoon that there could be as  \nmany as 25 dead. By early Wednesday, his deputies said the death \ntoll was 15, including the two gunmen.', 
                                        "The discrepancy occurred because the SWAT teams that picked  \ntheir way past bombs and bodies in an effort to secure building \ncovered overlapping areas, said sheriff's spokesman Steve Davis.",
                                        "``There were so many different SWAT teams in there, we were  \nconstantly getting different counts,'' Davis said.", 
                                        'As they gave periodic updates through the night, Davis and Stone  \nemphasized the death toll was unconfirmed. They said their priority \nwas making sure the school was safe.'])
        
        article_2 = Article(id='APW19990422.0095', date=datetime.date(1999, 4, 22), type=None, headline=None, 
                            paragraphs=['BURBANK, Calif. (AP) -- Republican presidential candidate Pat  \nBuchanan says stricter gun laws could not have prevented the deadly \nschool shootings in Littleton, Colo.', 
                                        "``The question is who has the weapons, the good people or those  \nwho are ugly and warped,'' he said. ``The problem began long before \nthey walked into school.''", 
                                        'Fifteen people, including the two killers, died Tuesday in a  \nshooting and bombing spree at Columbine High School.', 
                                        "``The massacre is a tragic reflection of the dark side of  \nAmerican society,'' Buchanan told reporters Wednesday. ``At \nLittleton, America got a glimpse of the last stop on that train to \nhell America boarded decades ago when we declared that God is dead \nand that each of us is his or her own god who can make up the rules \nas we go along.''"])
        
        article_3 = Article(id='APW19990427.0078', date=datetime.date(1999, 4, 27), type=None, headline=None, 
                            paragraphs=['LITTLETON, Colo. (AP) -- The day that Columbine High School  \nstudents are to return to class has been delayed because so many \nhave been attending funerals for students killed in the April 20 \nmassacre, an administrator said Tuesday.', 
                            'The students are scheduled to begin classes Monday at another  \nschool a few miles away in afternoon sessions until the end of the \nyear, said Barbara Monseu, area administrator for the Jefferson \nCounty School District.', 
                            "Students were originally scheduled to go back Thursday. Calling the return to the classroom ``a very important next step  \nin the process of healing,'' Monseu said the first thing Columbine \nstudents will do when they arrive at Chatfield High School is \nattend an assembly to reunite them with their teachers.", 
                            "``The teachers are very anxious to see their students again,''  \nshe said.", 
                            'Columbine and Chatfield are sports rivals, but junior John Danos  \nsaid he welcomed the newcomers.', 
                            "``I'm fine with it,'' he said. ``We're going to treat them  \nnormal.''", 
                            'To accommodate the 1,965 Columbine students, the school day is  \nbeing split with Chatfield students beginning early in the day and \nColumbine students showing up shortly before 1 p.m.', 
                            "To prepare for the resumption of classes, some of Columbine's  \n158 teachers met Tuesday morning for the first time since two \nstudent gunmen fatally shot 12 fellow students and a teacher before \ntaking their own lives."])
        
        article_4 = Article(id='APW19990422.0095', date=datetime.date(1999, 4, 22), type=None, headline=None, 
                            paragraphs=["Tuesday morning 12 columbine high school students and a teacher were murdered when eric harris and dylan klebold, also opened fire with at least four guns and dozens of bombs.",
                            "Jefferson county school officials said columbine's 1,800 students would return to classes thursday a few miles south at chatfield high school originally to columbine's overflow."])
        

        topic = Topic(topic_metadata, [article_1, article_2, article_3, article_4])
        document_group = DocumentGroup(topic)
        article_1 = document_group.articles[0]
        article_2 = document_group.articles[1]
        article_3 = document_group.articles[2]
        article_4 = document_group.articles[3]

        self.sent_a = Content(list(article_1.paragraphs[0].sents)[0], 1, article_1) # starting sent: date 4-21-1999, sent index 0
        self.sent_b = Content(list(article_2.paragraphs[0].sents)[0], 1, article_2) # 4-22-1999 A, sent index 0
        self.sent_c = Content(list(article_2.paragraphs[1].sents)[0], 1, article_2) # 4-22-1999 A, sent index 1
        self.sent_d = Content(list(article_3.paragraphs[2].sents)[1], 1, article_3) # 4-27-1999, sent index 3
        self.sent_e = Content(list(article_1.paragraphs[3].sents)[0], 1, article_1) # same article, sent index 4
        self.sent_f = Content(list(article_4.paragraphs[1].sents)[0], 1, article_4) # 4-22-1999 B, sent index 1

        realized_content = [self.sent_c, self.sent_d, self.sent_a, self.sent_b, self.sent_e, self.sent_f]
        realization_obj = Realization(selected_content=realized_content, doc_group=document_group, realized_content=realized_content)
        self.realization_obj = realization_obj
        self.ordered_content = Ordering(realization_obj)


    def test_sentence_indices(self):
        self.assertEqual(self.sent_a.span._.sent_index, 0)
        self.assertEqual(self.sent_e.span._.sent_index, 4)
        self.assertEqual(self.sent_a.article, self.sent_e.article)

        self.assertEqual(self.sent_b.span._.sent_index, 0)
        self.assertEqual(self.sent_c.span._.sent_index, 1)
        self.assertEqual(self.sent_d.span._.sent_index, 3)
        self.assertEqual(self.sent_f.span._.sent_index, 1)


    def test_select_first_sentence(self):
        first_sent = self.ordered_content.choose_starting_sentence(self.realization_obj.realized_content)
        self.assertEqual(first_sent, self.sent_a)


    def test_chronological_ordering_1(self):
        sents = self.realization_obj.realized_content.copy()
        first_sent = self.sent_a
        sents.remove(first_sent)

        next_sent = self.ordered_content.select_next_sentence(first_sent, sents, 0, 0, 1)
        self.assertEqual(next_sent, self.sent_e)

        sents.remove(next_sent)
        next_sent_2 = self.ordered_content.select_next_sentence(next_sent, sents, 0, 0, 1)
        self.assertEqual(next_sent_2, self.sent_b)

        sents.remove(next_sent_2)
        next_sent_3 = self.ordered_content.select_next_sentence(next_sent_2, sents, 0, 0, 1)
        self.assertEqual(next_sent_3, self.sent_c)

        sents.remove(next_sent_3)
        next_sent_4 = self.ordered_content.select_next_sentence(next_sent_3, sents, 0, 0, 1)
        self.assertEqual(next_sent_4, self.sent_f)

        sents.remove(next_sent_4)
        next_sent_5 = self.ordered_content.select_next_sentence(next_sent_4, sents, 0, 0, 1)
        self.assertEqual(next_sent_5, self.sent_d)


if __name__ == '__main__':
    unittest.main()
