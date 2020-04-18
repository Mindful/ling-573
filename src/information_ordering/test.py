import unittest
from collections import namedtuple
import spacy
from information_ordering.ordering import Ordering, is_redundant

SelectedContent = namedtuple('SelectedContent', ['doc_group', 'selected_content'])
Content = namedtuple('Content', ['span', 'weight'])
nlp = spacy.load("en_core_web_lg")

class TestOrdering(unittest.TestCase):
    def setUp(self):
        selected_content = {"APW19990911.0145": [Content(span=nlp("THE VALLEY, Anguilla (AP)")[:], weight=None), 
                                                Content(span=nlp("Hurricane Floyd on Saturday whirred away from the Caribbean's Leeward Islands, but forecasters warned that Puerto Rico and the Virgin Islands could be hit by heavy rains and flooding.")[:], weight=None), 
                                                Content(span=nlp("The hurricane itself was likely to pass well north of the islands, meteorologists said.")[:], weight=None),
                                                Content(span=nlp("Floyd was expected to intensify, and  it was unlikely to hit land soon.")[:], weight=None),
                                                Content(span=nlp("``But it's not going to threaten any land anywhere over the weekend,'' said Stacy Stewart, a meteorologist at the National Hurricane Center.")[:], weight=None),
                                                Content(span=nlp("It was too early to tell if the storm would threaten the Bahamas, which lie far north of other Caribbean islands.")[:], weight=None),
                                                Content(span=nlp("Floyd was expected to be east of the Bahamas in about three days.")[:], weight=None)],
                            "APW19990913.0046": [Content(span=nlp("MIAMI (AP) --")[:], weight=None),
                                                Content(span=nlp("As Hurricane Floyd neared the strength of a Category 5 storm today and headed west, thousands of coastal residents were ordered to evacuate and NASA began moving workers out of the low-lying Kennedy Space Center.")[:], weight=None),
                                                Content(span=nlp("``We need people to get ready right now, not tonight, not tomorrow,'' said Joan Heller, a spokeswoman for Brevard County, which ordered tens of thousands of people living in mobile homes and coastal areas to evacuate beginning at 4 p.m. today.")[:], weight=None),
                                                Content(span=nlp("``There are different tracks that have been projected and we don't come out unscathed on any of them,'' Heller said. ")[:], weight=None),
                                                Content(span=nlp("Gov. Jeb Bush, whose family lived in Miami during Hurricane Andrew in 1992, declared a state of emergency.")[:], weight=None),
                                                Content(span=nlp("That gives him the authority to deploy the state's National Guard and allows him to seize property, order evacuations and suspend tolls on the highways.")[:], weight=None)
]
        }
        self.selected_content = SelectedContent(doc_group=None, selected_content=selected_content)


    def test_chrological_order(self):
        ordering = Ordering(self.selected_content)
        first_sent = ordering.ordered_sents[0]
        last_sent = ordering.ordered_sents[-1]
        
        self.assertEqual(first_sent.span.text, "THE VALLEY, Anguilla (AP)")
        self.assertEqual(last_sent.span.text, "That gives him the authority to deploy the state's National Guard and allows him to seize property, order evacuations and suspend tolls on the highways.")


    def test_identify_redundant(self):
        content_obj = Content(span=nlp("That gives him the authority to deploy the state's National Guard and allows him to seize property, order evacuations and suspend tolls on the highways.")[:], weight=None)
        content_obj_2 = Content(span=nlp("That gives him the authority to deploy the state's National Guard and allows him to seize property, order evacuations and suspend tolls on the highways.")[:], weight=None)
        redundant = is_redundant(content_obj.span, content_obj_2.span)
        self.assertEqual(redundant, True)


    def test_identify_redundant_2(self):
        content_obj = Content(span=nlp("Bahamas residents abandoned beachfront homes and scrambled for emergency supplies today as Hurricane Floyd's 155 mph winds headed toward the vulnerable archipelago.")[:], weight=None)
        content_obj_2 = Content(span=nlp("Hurricane Floyd strengthened to a very dangerous Category 4 storm today with 155 mph wind, surprising forecasters and charging toward the Bahamas on a path that also threatened the Florida coast.")[:], weight=None)
        redundant = is_redundant(content_obj.span, content_obj_2.span)
        # default sim score of 0.9154215
        self.assertEqual(redundant, True)


    def test_identify_redundant_3(self):
        content_obj = Content(span=nlp("Bahamas residents abandoned beachfront homes and scrambled for emergency supplies today as Hurricane Floyd's 155 mph winds headed toward the vulnerable archipelago.")[:], weight=None)
        content_obj_2 = Content(span=nlp("Heavy rain started falling in parts of the Bahamas today as residents scrambled for emergency supplies, shelters opened, schools closed and shop owners boarded up windows.")[:], weight=None)
        redundant = is_redundant(content_obj.span, content_obj_2.span)
        # default sim score of 0.88998574
        self.assertEqual(redundant, True)


    def test_identify_redundant_4(self):
        content_obj = Content(span=nlp("It was headed west at nearly 15 mph, and was expected to gradually turn to a west-northwesterly-heading by evening, with a further turn toward the northwest on Tuesday.")[:], weight=None)
        content_obj_2 = Content(span=nlp("Floyd was expected to intensify, and it was unlikely to hit land soon.")[:], weight=None)
        redundant = is_redundant(content_obj.span, content_obj_2.span)
        # default sim score of 0.8860523
        self.assertEqual(redundant, True)


    def test_identify_redundant_5(self):
        content_obj = Content(span=nlp("Bahamas residents abandoned beachfront homes and scrambled for emergency supplies today as Hurricane Floyd's 155 mph winds headed toward the vulnerable archipelago.")[:], weight=None)
        content_obj_2 = Content(span=nlp("NASA's four space shuttles were in danger as the agency braced Monday for Hurricane Floyd, a storm powerful enough to wipe out its launch pads and hangars.")[:], weight=None)
        redundant = is_redundant(content_obj.span, content_obj_2.span)
        # default sim score of 0.8578921
        self.assertEqual(redundant, False)


    def test_identify_redundant_6(self):
        content_obj = Content(span=nlp("Bahamas residents abandoned beachfront homes and scrambled for emergency supplies today as Hurricane Floyd's 155 mph winds headed toward the vulnerable archipelago.")[:], weight=None)
        content_obj_2 = Content(span=nlp("Hurricane Floyd continued to push toward Florida on Monday, prompting evacuation orders for coastal residents.")[:], weight=None)
        redundant = is_redundant(content_obj.span, content_obj_2.span)
        # default sim score of 0.90372616
        self.assertEqual(redundant, True)


    def test_identify_redundant_7(self):
        content_obj = Content(span=nlp("As Hurricane Floyd neared the strength of a Category 5 storm today and headed west, thousands of coastal residents were ordered to evacuate and NASA began moving workers out of the low-lying Kennedy Space Center.")[:], weight=None)
        content_obj_2 = Content(span=nlp("NASA's four space shuttles were in danger as the agency braced Monday for Hurricane Floyd, a storm powerful enough to wipe out its launch pads and hangars.")[:], weight=None)
        redundant = is_redundant(content_obj.span, content_obj_2.span)
        # default sim score of 0.9321409
        self.assertEqual(redundant, True)


    def test_identify_redundant_8(self):
        content_obj = Content(span=nlp("Floyd was moving due west at nearly 14 mph today, but was expected to begin gradually turning by evening, initially taking a west-northwest course.")[:], weight=None)
        content_obj_2 = Content(span=nlp("A ridge of high pressure has been blocking Floyd and forcing it to move almost due west, but a low-pressure system moving east could weaken the high pressure.")[:], weight=None)
        redundant = is_redundant(content_obj.span, content_obj_2.span)
        # default sim score of 0.8964482
        self.assertEqual(redundant, True)


    def test_identify_redundant_9(self):
        content_obj = Content(span=nlp("Floyd was moving due west at nearly 14 mph today, but was expected to begin gradually turning by evening, initially taking a west-northwest course.")[:], weight=None)
        content_obj_2 = Content(span=nlp("In other markets, orange juice futures rose as Hurricane Floyd bore down on the southeastern U.S. coastline, and platinum futures advanced sharply.")[:], weight=None)
        redundant = is_redundant(content_obj.span, content_obj_2.span)
        # default sim score of 0.8156167
        self.assertEqual(redundant, False)


if __name__ == '__main__':
    unittest.main()
