import os
from pathlib import Path
import xml.etree.cElementTree as ET
import xml.dom.minidom

'''
Generates a ROUGE config file that can be used to run the ROUGE calculation,
given the directories for 1. The summary outputs (peer_root_dir) and 2. The human model summaries (model_dir)

Example usage:
    output_rouge_config('/573/Data/mydata', '/573/Data/models/devtest/')
'''


def output_rouge_config(peer_root_dir, model_dir, output_dir='output', out_filename='rouge_config'):
    rouge_eval = ET.Element("ROUGE_EVAL", version="1.5.5")
    eval_groups = [file_name for file_name in os.listdir(peer_root_dir)]

    for eval_group in sorted(eval_groups):
        _add_eval_group(rouge_eval, eval_group, peer_root_dir, model_dir)

    tree = ET.ElementTree(rouge_eval)
    _write_config_file(output_dir, out_filename, tree)


def _add_eval_group(rouge_eval, eval_group, peer_root_dir, model_dir):
    eval_group_id = eval_group[:-2]
    eval_elem = ET.SubElement(rouge_eval, "EVAL", ID=eval_group_id)
    ET.SubElement(eval_elem, "PEER-ROOT").text = peer_root_dir
    ET.SubElement(eval_elem, "MODEL-ROOT").text = model_dir
    ET.SubElement(eval_elem, "INPUT-FORMAT", TYPE="SPL")

    peers_element = ET.SubElement(eval_elem, "PEERS")
    p_id = _get_last_segement(eval_group)
    ET.SubElement(peers_element, "P", ID=p_id).text = eval_group

    models_elem = ET.SubElement(eval_elem, "MODELS")
    _add_models(models_elem, eval_elem, eval_group_id, model_dir)


def _add_models(models_elem, eval_elem, eval_group_id, model_dir):
    eval_model_files = [
        file_name for file_name in os.listdir(model_dir)
        if eval_group_id in file_name
    ]

    for model_file in sorted(eval_model_files):
        m_id = _get_last_segement(model_file)
        ET.SubElement(models_elem, "M", ID=m_id).text = model_file


def _get_last_segement(text):
    return text[text.rindex('.') +1:]


def _write_config_file(output_dir, out_filename, tree):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    file_name = "{}/{}.xml".format(output_dir, out_filename)
    tree.write(file_name)

    # pretty print, instead of all one line
    dom = xml.dom.minidom.parse(file_name)
    pretty_xml = dom.toprettyxml()
    with open(file_name, 'w') as config_file:
        config_file.write(pretty_xml)

