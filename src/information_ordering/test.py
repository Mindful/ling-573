import unittest
from collections import namedtuple
import spacy
from information_ordering.ordering import Ordering

SelectedContext = namedtuple('SelectedContent', ['doc_group', 'selected_content'])
nlp = spacy.load("en_core_web_lg")

class TestOrdering(unittest.TestCase):
    def setUp(self):
        selected_content = {"APW19990911.0145": [nlp("THE VALLEY, Anguilla (AP)"), 
                                                nlp("Hurricane Floyd on Saturday whirred away from the Caribbean's Leeward Islands, but forecasters warned that Puerto Rico and the Virgin Islands could be hit by heavy rains and flooding."), 
                                                nlp("The hurricane itself was likely to pass well north of the islands, meteorologists said."),
                                                nlp("Floyd was expected to intensify, and  it was unlikely to hit land soon."),
                                                nlp("``But it's not going to threaten any land anywhere over the weekend,'' said Stacy Stewart, a meteorologist at the National Hurricane Center."),
                                                nlp("It was too early to tell if the storm would threaten the Bahamas, which lie far north of other Caribbean islands."),
                                                nlp("Floyd was expected to be east of the Bahamas in about three days.")],
                            "APW19990913.0046": [nlp("MIAMI (AP) --"),
                                                nlp("As Hurricane Floyd neared the strength of a Category 5 storm today and headed west, thousands of coastal residents were ordered to evacuate and NASA began moving workers out of the low-lying Kennedy Space Center."),
                                                nlp("``We need people to get ready right now, not tonight, not tomorrow,'' said Joan Heller, a spokeswoman for Brevard County, which ordered tens of thousands of people living in mobile homes and coastal areas to evacuate beginning at 4 p.m. today."),
                                                nlp("``There are different tracks that have been projected and we don't come out unscathed on any of them,'' Heller said. "),
                                                nlp("Gov. Jeb Bush, whose family lived in Miami during Hurricane Andrew in 1992, declared a state of emergency."),
                                                nlp("That gives him the authority to deploy the state's National Guard and allows him to seize property, order evacuations and suspend tolls on the highways.")
]
        }
        self.selected_content = SelectedContext(doc_group=None, selected_content=selected_content)


    def test_chrological_order(self):
        ordering = Ordering(self.selected_content)
        first_sent = ordering.ordered_sents[0]
        last_sent = ordering.ordered_sents[-1]
        
        self.assertEqual(first_sent.text, "THE VALLEY, Anguilla (AP)")
        self.assertEqual(last_sent.text, "That gives him the authority to deploy the state's National Guard and allows him to seize property, order evacuations and suspend tolls on the highways.")



if __name__ == '__main__':
    unittest.main()
