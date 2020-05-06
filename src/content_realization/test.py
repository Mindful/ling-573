import unittest
from collections import namedtuple
from content_realization.realization import is_redundant
from common import NLP

Content = namedtuple('Content', ['span', 'weight'])

class TestContentRealization(unittest.TestCase):
    def test_identify_redundant(self):
        content_obj = Content(span=NLP("That gives him the authority to deploy the state's National Guard and allows him to seize property, order evacuations and suspend tolls on the highways.")[:], weight=None)
        content_obj_2 = Content(span=NLP("That gives him the authority to deploy the state's National Guard and allows him to seize property, order evacuations and suspend tolls on the highways.")[:], weight=None)
        redundant = is_redundant(content_obj.span, content_obj_2.span)
        self.assertEqual(redundant, True)


    def test_identify_redundant_2(self):
        content_obj = Content(span=NLP("Bahamas residents abandoned beachfront homes and scrambled for emergency supplies today as Hurricane Floyd's 155 mph winds headed toward the vulnerable archipelago.")[:], weight=None)
        content_obj_2 = Content(span=NLP("Hurricane Floyd strengthened to a very dangerous Category 4 storm today with 155 mph wind, surprising forecasters and charging toward the Bahamas on a path that also threatened the Florida coast.")[:], weight=None)
        redundant = is_redundant(content_obj.span, content_obj_2.span)
        # default sim score of 0.9154215
        self.assertEqual(redundant, True)


    def test_identify_redundant_3(self):
        content_obj = Content(span=NLP("Bahamas residents abandoned beachfront homes and scrambled for emergency supplies today as Hurricane Floyd's 155 mph winds headed toward the vulnerable archipelago.")[:], weight=None)
        content_obj_2 = Content(span=NLP("Heavy rain started falling in parts of the Bahamas today as residents scrambled for emergency supplies, shelters opened, schools closed and shop owners boarded up windows.")[:], weight=None)
        redundant = is_redundant(content_obj.span, content_obj_2.span)
        # default sim score of 0.88998574
        self.assertEqual(redundant, True)


    def test_identify_redundant_4(self):
        content_obj = Content(span=NLP("It was headed west at nearly 15 mph, and was expected to gradually turn to a west-northwesterly-heading by evening, with a further turn toward the northwest on Tuesday.")[:], weight=None)
        content_obj_2 = Content(span=NLP("Floyd was expected to intensify, and it was unlikely to hit land soon.")[:], weight=None)
        redundant = is_redundant(content_obj.span, content_obj_2.span)
        # default sim score of 0.8860523
        self.assertEqual(redundant, True)


    def test_identify_redundant_5(self):
        content_obj = Content(span=NLP("Bahamas residents abandoned beachfront homes and scrambled for emergency supplies today as Hurricane Floyd's 155 mph winds headed toward the vulnerable archipelago.")[:], weight=None)
        content_obj_2 = Content(span=NLP("NASA's four space shuttles were in danger as the agency braced Monday for Hurricane Floyd, a storm powerful enough to wipe out its launch pads and hangars.")[:], weight=None)
        redundant = is_redundant(content_obj.span, content_obj_2.span)
        # default sim score of 0.8578921
        self.assertEqual(redundant, False)


    def test_identify_redundant_6(self):
        content_obj = Content(span=NLP("Bahamas residents abandoned beachfront homes and scrambled for emergency supplies today as Hurricane Floyd's 155 mph winds headed toward the vulnerable archipelago.")[:], weight=None)
        content_obj_2 = Content(span=NLP("Hurricane Floyd continued to push toward Florida on Monday, prompting evacuation orders for coastal residents.")[:], weight=None)
        redundant = is_redundant(content_obj.span, content_obj_2.span)
        # default sim score of 0.90372616
        self.assertEqual(redundant, True)


    def test_identify_redundant_7(self):
        content_obj = Content(span=NLP("As Hurricane Floyd neared the strength of a Category 5 storm today and headed west, thousands of coastal residents were ordered to evacuate and NASA began moving workers out of the low-lying Kennedy Space Center.")[:], weight=None)
        content_obj_2 = Content(span=NLP("NASA's four space shuttles were in danger as the agency braced Monday for Hurricane Floyd, a storm powerful enough to wipe out its launch pads and hangars.")[:], weight=None)
        redundant = is_redundant(content_obj.span, content_obj_2.span)
        # default sim score of 0.9321409
        self.assertEqual(redundant, True)


    def test_identify_redundant_8(self):
        content_obj = Content(span=NLP("Floyd was moving due west at nearly 14 mph today, but was expected to begin gradually turning by evening, initially taking a west-northwest course.")[:], weight=None)
        content_obj_2 = Content(span=NLP("A ridge of high pressure has been blocking Floyd and forcing it to move almost due west, but a low-pressure system moving east could weaken the high pressure.")[:], weight=None)
        redundant = is_redundant(content_obj.span, content_obj_2.span)
        # default sim score of 0.8964482
        self.assertEqual(redundant, True)


    def test_identify_redundant_9(self):
        content_obj = Content(span=NLP("Floyd was moving due west at nearly 14 mph today, but was expected to begin gradually turning by evening, initially taking a west-northwest course.")[:], weight=None)
        content_obj_2 = Content(span=NLP("In other markets, orange juice futures rose as Hurricane Floyd bore down on the southeastern U.S. coastline, and platinum futures advanced sharply.")[:], weight=None)
        redundant = is_redundant(content_obj.span, content_obj_2.span)
        # default sim score of 0.8156167
        self.assertEqual(redundant, False)


if __name__ == '__main__':
    unittest.main()
