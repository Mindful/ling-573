import unittest
from collections import namedtuple
import spacy
from information_ordering.ordering import Ordering
from common import NLP

SelectedContent = namedtuple('SelectedContent', ['doc_group', 'selected_content'])
Content = namedtuple('Content', ['span', 'weight'])

class TestOrdering(unittest.TestCase):
    def setUp(self):
        selected_content = {"APW19990911.0145": [Content(span=NLP("THE VALLEY, Anguilla (AP)")[:], weight=None), 
                                                Content(span=NLP("Hurricane Floyd on Saturday whirred away from the Caribbean's Leeward Islands, but forecasters warned that Puerto Rico and the Virgin Islands could be hit by heavy rains and flooding.")[:], weight=None), 
                                                Content(span=NLP("The hurricane itself was likely to pass well north of the islands, meteorologists said.")[:], weight=None),
                                                Content(span=NLP("Floyd was expected to intensify, and  it was unlikely to hit land soon.")[:], weight=None),
                                                Content(span=NLP("``But it's not going to threaten any land anywhere over the weekend,'' said Stacy Stewart, a meteorologist at the National Hurricane Center.")[:], weight=None),
                                                Content(span=NLP("It was too early to tell if the storm would threaten the Bahamas, which lie far north of other Caribbean islands.")[:], weight=None),
                                                Content(span=NLP("Floyd was expected to be east of the Bahamas in about three days.")[:], weight=None)],
                            "APW19990913.0046": [Content(span=NLP("MIAMI (AP) --")[:], weight=None),
                                                Content(span=NLP("As Hurricane Floyd neared the strength of a Category 5 storm today and headed west, thousands of coastal residents were ordered to evacuate and NASA began moving workers out of the low-lying Kennedy Space Center.")[:], weight=None),
                                                Content(span=NLP("``We need people to get ready right now, not tonight, not tomorrow,'' said Joan Heller, a spokeswoman for Brevard County, which ordered tens of thousands of people living in mobile homes and coastal areas to evacuate beginning at 4 p.m. today.")[:], weight=None),
                                                Content(span=NLP("``There are different tracks that have been projected and we don't come out unscathed on any of them,'' Heller said. ")[:], weight=None),
                                                Content(span=NLP("Gov. Jeb Bush, whose family lived in Miami during Hurricane Andrew in 1992, declared a state of emergency.")[:], weight=None),
                                                Content(span=NLP("That gives him the authority to deploy the state's National Guard and allows him to seize property, order evacuations and suspend tolls on the highways.")[:], weight=None)
]
        }
        self.selected_content = SelectedContent(doc_group=None, selected_content=selected_content)


if __name__ == '__main__':
    unittest.main()
