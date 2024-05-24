import json

from file_paths import name_att_rel_count_fpath

name_att_rel_count = json.load(open(name_att_rel_count_fpath, "r"))

class PhraseHandler(object):
    def __init__(self, word_embed=None, phrase_length=10,
                cat_count_thresh=21, att_count_thresh=21, rel_count_thresh=21):
        self.word_embed = word_embed
        self.phrase_length = phrase_length
        
        # Convert to dict
        def convert_to_dict(item):
            if isinstance(item, list):
                return dict(item)
            return item
                
        # Load cat/att/rel count data
        self.cat_to_count_dict = convert_to_dict(name_att_rel_count['cat'])
        self.att_to_count_dict = convert_to_dict(name_att_rel_count['att'])
        self.rel_to_count_dict = convert_to_dict(name_att_rel_count['rel'])

        # Prepare cat/att/rel
         # Category
        self.cat_to_count = {k: c for (k, c) in self.cat_to_count_dict}
        self.cat_to_count['[INV]'] = 0
        self.cat_to_count['[UNK]'] = 0
        self.label_to_cat = ['[INV]'] + [k for (k, c) in self.cat_to_count_dict if c >= cat_count_thresh] + ['[UNK]']
        self.cat_to_label = {cat: l for l, cat in enumerate(self.label_to_cat)}
        print('Number of relationships: %d / %d, frequency thresh: %d (excluding [INV] [UNK])'
             % (len(self.label_to_cat) -2, len(self.cat_to_count_dict), cat_count_thresh))
         # Attributes
        self.att_to_count = {k: c for (k, c) in self.att_to_count_dict}
        self.att_to_count['[INV]'] = 0
        self.att_to_count['[UNK]'] = 0
        self.label_to_att = ['[INV]'] + [k for (k, c) in self.att_to_count_dict if c >= att_count_thresh] + ['[UNK]']
        self.att_to_label = {att: l for l, att in enumerate(self.label_to_att)}
        print('Number of relationships: %d / %d, frequency thresh: %d (excluding [INV] [UNK])'
             % (len(self.label_to_att) -2, len(self.att_to_count_dict), att_count_thresh))
         # Relationships
        self.label_to_rel = ['[INV]'] + [k for (k, c) in self.rel_to_count_dict if c >= rel_count_thresh] + ['[UNK]']
        self.rel_to_label = {rel: l for l, rel in enumerate(self.rel_to_att)}
        print('Number of relationships: %d / %d, frequency thresh: %d (excluding [INV] [UNK])'
             % (len(self.label_to_rel) -2, len(self.rel_to_count_dict), rel_count_thresh))
            