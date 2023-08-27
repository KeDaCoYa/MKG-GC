
import json
import pickle


def load_cache_dictionary(cahe_path):
    """
    加载已经经过
    """
    with open(cahe_path, 'rb') as fin:
        cached_dictionary = pickle.load(fin)


    dictionary, dict_sparse_embeds, dict_dense_embeds = (
        cached_dictionary['dictionary'],
        cached_dictionary['dict_sparse_embeds'],
        cached_dictionary['dict_dense_embeds'],
    )
    return dictionary, dict_sparse_embeds, dict_dense_embeds


def read_raw_dataset(file_path,type='disease'):
    """
    读取ner的结果，对ner的结果进行标准化
    但是会筛选一下针对的实体类别，
    :param file_path:
    :param type: all,disease,chemical,protein,gene,cell,....
    :return:
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    abstract_ids = list(all_data.keys())

    for id in abstract_ids:

        abstract_data = all_data[id]
        abstract_sentence_li = abstract_data['abstract_sentence_li']
        entities = abstract_data['entities']

        for ent in entities:
            ent_name = ent['entity_name']
            ent_start_idx = int(ent['start_idx'])
            ent_end_idx = int(ent['end_idx'])
            ent_type = ent['entity_type']
            ent_pos = int(ent['pos'])
            ent_id = ent['id']



def return_dictionary_url(norm_id,entity_type):
    if entity_type in ['Gene/Protein','DNA','RNA']:
        url_pattern = "https://www.ncbi.nlm.nih.gov/gene/{}"
        _,id_ = norm_id.split(':')
        url = url_pattern.format(id_)
        return url
    elif entity_type == 'Disease':
        url = "https://meshb-prev.nlm.nih.gov/record/ui?ui={}".format(norm_id)
    elif entity_type == 'cell_line':
        url = "https://web.expasy.org/cellosaurus/{}".format(norm_id)
    elif entity_type == 'cell_type':
        url = "http://purl.obolibrary.org/obo/{}".format(norm_id)
    elif entity_type == 'Chemical/Drug':
        url = "https://meshb-prev.nlm.nih.gov/record/ui?ui={}".format(norm_id)
    elif entity_type == 'Species':
        url = "https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?mode=Info&id={}".format(norm_id[:-2])
    else:
        raise ValueError("no this entity type:{}".format(entity_type))
    return url
if __name__ == '__main__':
    read_raw_dataset("/root/code/bioner/BioNormalization/dataset/abstract_res/single_model_abstracts_entities.json")







